using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading;
using System.Xml.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Gym.SourceGenerators {
    /// <summary>
    ///     Gym.NET's source generator. For every class marked <c>[GymOptions]</c> (DR-34) it writes a <c>&lt;Class&gt;Options</c>
    ///     record from the class's constructor and a constructor that takes it; for every record marked <c>[GymTuple]</c> or
    ///     <c>[GymDict]</c> (DR-13) it writes the conversions to and from the generic form, Python's <c>==</c> and Python's
    ///     <c>repr</c>. Generating from the code itself keeps the records in step with the constructors they describe.
    /// </summary>
    /// <remarks>
    ///     Each output is pure text computed from the symbol, so the incremental pipeline caches it by value and regenerates
    ///     only the classes that changed. A class that is not <c>partial</c> fails to compile with the compiler's own error
    ///     (CS0260), which names the class, so no diagnostic of the generator's own is needed.
    /// </remarks>
    [Generator(LanguageNames.CSharp)]
    public sealed class GymRecordsGenerator : IIncrementalGenerator {
        /// <summary>The attribute that asks for an options record (DR-34).</summary>
        private const string OptionsAttribute = "Gym.Utils.GymOptionsAttribute";

        /// <summary>The attribute that makes a positional record a Python tuple (DR-13).</summary>
        private const string TupleAttribute = "Gym.Utils.GymTupleAttribute";

        /// <summary>The attribute that makes a positional record a Python dict (DR-13).</summary>
        private const string DictAttribute = "Gym.Utils.GymDictAttribute";

        /// <summary>The attribute that names a parameter or property by its Python name.</summary>
        private const string PyNameAttribute = "Gym.Utils.PyNameAttribute";

        /// <summary>Types written fully qualified, special types by their keyword, nullable annotations kept.</summary>
        private static readonly SymbolDisplayFormat TypeFormat =
            SymbolDisplayFormat.FullyQualifiedFormat.AddMiscellaneousOptions(SymbolDisplayMiscellaneousOptions.IncludeNullableReferenceTypeModifier);

        /// <summary>Wires the three attribute pipelines to their emitters.</summary>
        /// <param name="context">The generator's initialization context.</param>
        public void Initialize(IncrementalGeneratorInitializationContext context) {
            IncrementalValuesProvider<GeneratedFile?> options = context.SyntaxProvider.ForAttributeWithMetadataName(OptionsAttribute,
                static (node, _) => node is ClassDeclarationSyntax,
                static (attributeContext, cancellationToken) => OptionsRecord(attributeContext, cancellationToken));
            IncrementalValuesProvider<GeneratedFile?> tuples = context.SyntaxProvider.ForAttributeWithMetadataName(TupleAttribute,
                static (node, _) => node is RecordDeclarationSyntax,
                static (attributeContext, cancellationToken) => CollectionRecord(attributeContext, asDict: false, cancellationToken));
            IncrementalValuesProvider<GeneratedFile?> dicts = context.SyntaxProvider.ForAttributeWithMetadataName(DictAttribute,
                static (node, _) => node is RecordDeclarationSyntax,
                static (attributeContext, cancellationToken) => CollectionRecord(attributeContext, asDict: true, cancellationToken));

            context.RegisterSourceOutput(options, Emit);
            context.RegisterSourceOutput(tuples, Emit);
            context.RegisterSourceOutput(dicts, Emit);
        }

        /// <summary>Adds one generated file, when the transform produced one.</summary>
        /// <param name="context">The output context.</param>
        /// <param name="file">The file, or <see langword="null"/> for a target that cannot be generated for.</param>
        private static void Emit(SourceProductionContext context, GeneratedFile? file) {
            if (file is not null)
                context.AddSource(file.HintName, file.Source);
        }

        /// <summary>
        ///     The options record of a <c>[GymOptions]</c> class: one property per constructor parameter (a wrapper's
        ///     <c>env</c> excepted), each remembering whether it was set, <c>ToKwargs()</c> holding the set ones by Python name,
        ///     and a constructor of the class that takes the record.
        /// </summary>
        /// <param name="attributeContext">The attributed class.</param>
        /// <param name="cancellationToken">Cancels the transform.</param>
        /// <returns>The file, or <see langword="null"/> for a class without a usable public constructor.</returns>
        private static GeneratedFile? OptionsRecord(GeneratorAttributeSyntaxContext attributeContext, CancellationToken cancellationToken) {
            if (attributeContext.TargetSymbol is not INamedTypeSymbol type)
                return null;

            // The constructor make binds kwargs to first: the public one with the most parameters, without ref, out or params.
            IMethodSymbol? constructor = type.InstanceConstructors
                .Where(c => c.DeclaredAccessibility == Accessibility.Public && c.Parameters.All(p => p.RefKind == RefKind.None && !p.IsParams))
                .Where(c => !c.Parameters.Any(p => p.Type.Name == type.Name + "Options"))
                .OrderByDescending(c => c.Parameters.Length)
                .FirstOrDefault();
            if (constructor is null)
                return null;
            cancellationToken.ThrowIfCancellationRequested();

            IParameterSymbol[] options = constructor.Parameters.Where(p => p.Name != "env").ToArray();
            IParameterSymbol? env = constructor.Parameters.FirstOrDefault(p => p.Name == "env");
            Dictionary<string, string> parameterDocs = ParameterDocs(constructor);

            // The record is generic when an option's type mentions the class's type parameters (a TObs fill value).
            ITypeParameterSymbol[] usedTypeParameters = type.TypeParameters.Where(t => options.Any(p => Mentions(p.Type, t))).ToArray();
            string recordName = type.Name + "Options";
            string recordTypeArguments = usedTypeParameters.Length == 0 ? "" : "<" + string.Join(", ", usedTypeParameters.Select(t => t.Name)) + ">";
            string classTypeParameters = type.TypeParameters.Length == 0 ? "" : "<" + string.Join(", ", type.TypeParameters.Select(t => t.Name)) + ">";
            string accessibility = type.DeclaredAccessibility == Accessibility.Public ? "public" : "internal";

            var source = new StringBuilder();
            source.AppendLine("// <auto-generated/>");
            source.AppendLine("#nullable enable");
            string? ns = type.ContainingNamespace.IsGlobalNamespace ? null : type.ContainingNamespace.ToDisplayString();
            if (ns is not null)
                source.Append("namespace ").Append(ns).AppendLine(" {");

            source.AppendLine("    /// <summary>");
            source.Append("    ///     The keyword arguments of <see cref=\"").Append(DocCref(type)).AppendLine("\"/>'s constructor as a typed record (DR-34), generated");
            source.AppendLine("    ///     from the constructor so it cannot drift from it. A property that is set becomes a keyword argument by its Python");
            source.AppendLine("    ///     name; one left unset is not passed, so <c>make</c> keeps the registered spec's value and the constructor its default.");
            source.AppendLine("    /// </summary>");
            source.Append("    ").Append(accessibility).Append(" sealed partial record ").Append(recordName).Append(recordTypeArguments).AppendLine(" : global::Gym.Utils.IGymOptions {");

            foreach (IParameterSymbol parameter in options) {
                string property = PropertyName(parameter.Name);
                string field = "_" + char.ToLowerInvariant(property[0]) + property.Substring(1);
                string typeName = parameter.Type.ToDisplayString(TypeFormat);
                string initializer = parameter.HasExplicitDefaultValue ? " = " + Literal(parameter.ExplicitDefaultValue, parameter.Type) + ";" : " = default!;";
                source.Append("        /// <summary>The value of <see cref=\"").Append(property).AppendLine("\"/>, the constructor's default until set.</summary>");
                source.Append("        private ").Append(typeName).Append(' ').Append(field).AppendLine(initializer);
                source.Append("        /// <summary>Whether <see cref=\"").Append(property).AppendLine("\"/> was set, so it is passed on.</summary>");
                source.Append("        private bool ").Append(field).AppendLine("Set;");
                source.AppendLine("        /// <summary>");
                // The constructor's own <param> text, so the record documents what the constructor does with the value.
                if (parameterDocs.TryGetValue(parameter.Name, out string? documented))
                    source.Append("        ///     ").AppendLine(documented);
                source.Append("        ///     The <c>").Append(PythonName(parameter)).AppendLine("</c> keyword argument, passed on only once set.");
                source.AppendLine("        /// </summary>");
                source.Append("        public ").Append(typeName).Append(' ').Append(property).Append(" { get => ").Append(field).Append("; init { ").Append(field).Append(" = value; ")
                    .Append(field).AppendLine("Set = true; } }");
            }

            source.AppendLine("        /// <summary>The set properties as keyword arguments by Python name, in constructor order.</summary>");
            source.AppendLine("        /// <returns>A new dict the caller owns.</returns>");
            source.AppendLine("        public global::Gym.Envs.Info ToKwargs() {");
            source.AppendLine("            var kwargs = new global::Gym.Envs.Info();");
            foreach (IParameterSymbol parameter in options) {
                string property = PropertyName(parameter.Name);
                string field = "_" + char.ToLowerInvariant(property[0]) + property.Substring(1);
                source.Append("            if (").Append(field).Append("Set) kwargs[").Append(SymbolDisplay.FormatLiteral(PythonName(parameter), true)).Append("] = ").Append(field).AppendLine(";");
            }

            source.AppendLine("            return kwargs;");
            source.AppendLine("        }");
            source.AppendLine("    }");
            source.AppendLine();

            // The class's constructor from the record; required parameters left unset are Python's missing-argument error.
            string[] required = options.Where(p => !p.HasExplicitDefaultValue && !p.IsOptional).Select(p => PythonName(p)).ToArray();
            string recordType = recordName + recordTypeArguments;
            source.Append("    partial class ").Append(type.Name).Append(classTypeParameters).AppendLine(" {");
            source.AppendLine("        /// <summary>");
            source.Append("        ///     Builds the ").Append(type.TypeKind == TypeKind.Class ? "instance" : "value").Append(" from its options record: the properties set, the constructor's defaults for the rest.")
                .AppendLine();
            source.AppendLine("        /// </summary>");
            if (env is not null)
                source.AppendLine("        /// <param name=\"env\">The env to wrap.</param>");
            source.AppendLine("        /// <param name=\"options\">The options.</param>");
            source.AppendLine("        /// <exception cref=\"global::System.ArgumentNullException\">Thrown when <paramref name=\"options\"/> is <see langword=\"null\"/>.</exception>");
            if (required.Length > 0)
                source.AppendLine("        /// <exception cref=\"global::NumSharp.TypeError\">Thrown when a required argument is unset, with Python's missing-argument message.</exception>");
            source.Append("        public ").Append(type.Name).Append('(');
            if (env is not null)
                source.Append(env.Type.ToDisplayString(TypeFormat)).Append(" env, ");
            source.Append(recordType).Append(" options) : this(");
            string checkedOptions = $"__CheckOptions(options)";
            source.Append(string.Join(", ", constructor.Parameters.Select(p => p.Name == "env" ? "env" : $"{checkedOptions}.{PropertyName(p.Name)}")));
            source.AppendLine(") { }");
            source.AppendLine();
            source.AppendLine("        /// <summary>Refuses a missing record, and required arguments left unset, as Python refuses a call missing them.</summary>");
            source.AppendLine("        /// <param name=\"options\">The options.</param>");
            source.AppendLine("        /// <returns><paramref name=\"options\"/>.</returns>");
            source.AppendLine("        /// <exception cref=\"global::System.ArgumentNullException\">Thrown when <paramref name=\"options\"/> is <see langword=\"null\"/>.</exception>");
            if (required.Length > 0)
                source.AppendLine("        /// <exception cref=\"global::NumSharp.TypeError\">Thrown when a required argument is unset.</exception>");
            source.Append("        private static ").Append(recordType).Append(" __CheckOptions(").Append(recordType).AppendLine(" options) {");
            source.AppendLine("            if (options is null)");
            source.AppendLine("                throw new global::System.ArgumentNullException(nameof(options));");
            if (required.Length > 0) {
                source.AppendLine("            global::Gym.Envs.Info set = options.ToKwargs();");
                source.Append("            string[] missing = new[] { ").Append(string.Join(", ", required.Select(r => SymbolDisplay.FormatLiteral(r, true)))).AppendLine(" }");
                source.AppendLine("                .Where(name => !set.ContainsKey(name)).ToArray();");
                source.AppendLine("            if (missing.Length > 0) {");
                source.AppendLine("                string[] quoted = missing.Select(name => \"'\" + name + \"'\").ToArray();");
                source.AppendLine("                string names = quoted.Length == 1 ? quoted[0] : quoted.Length == 2 ? quoted[0] + \" and \" + quoted[1]");
                source.AppendLine("                    : string.Join(\", \", quoted, 0, quoted.Length - 1) + \", and \" + quoted[quoted.Length - 1];");
                source.Append("                throw new global::NumSharp.TypeError(\"").Append(type.Name)
                    .AppendLine(".__init__() missing \" + missing.Length + \" required positional argument\" + (missing.Length == 1 ? \"\" : \"s\") + \": \" + names);");
                source.AppendLine("            }");
            }

            source.AppendLine("            return options;");
            source.AppendLine("        }");
            source.AppendLine("    }");
            if (ns is not null)
                source.AppendLine("}");

            string text = source.ToString();
            if (required.Length > 0)
                text = text.Replace("#nullable enable", "#nullable enable\nusing System.Linq;");
            return new GeneratedFile($"{HintPrefix(type)}.Options.g.cs", text);
        }

        /// <summary>
        ///     The members of a <c>[GymTuple]</c> or <c>[GymDict]</c> positional record: implicit conversions to and from the
        ///     generic form (<see langword="object"/>?[] for a tuple, <c>Info</c> for a dict), Python's <c>==</c> through
        ///     <c>PyEquality</c>, a hash consistent with it, and Python's <c>repr</c>.
        /// </summary>
        /// <param name="attributeContext">The attributed record.</param>
        /// <param name="asDict">Whether the record is a dict (else a tuple).</param>
        /// <param name="cancellationToken">Cancels the transform.</param>
        /// <returns>The file, or <see langword="null"/> for a record without a primary constructor.</returns>
        private static GeneratedFile? CollectionRecord(GeneratorAttributeSyntaxContext attributeContext, bool asDict, CancellationToken cancellationToken) {
            if (attributeContext.TargetSymbol is not INamedTypeSymbol type || attributeContext.TargetNode is not RecordDeclarationSyntax declaration
                || declaration.ParameterList is null)
                return null;

            IMethodSymbol? primary = type.InstanceConstructors.FirstOrDefault(c => c.Parameters.Length == declaration.ParameterList.Parameters.Count
                                                                                  && c.Parameters.Select(p => p.Name).SequenceEqual(declaration.ParameterList.Parameters.Select(p => p.Identifier.ValueText)));
            if (primary is null)
                return null;
            cancellationToken.ThrowIfCancellationRequested();

            string typeParameters = type.TypeParameters.Length == 0 ? "" : "<" + string.Join(", ", type.TypeParameters.Select(t => t.Name)) + ">";
            string self = type.Name + typeParameters;
            string generic = asDict ? "global::Gym.Envs.Info" : "object?[]";
            bool isSealed = type.IsSealed;
            IParameterSymbol[] items = primary.Parameters.ToArray();

            var source = new StringBuilder();
            source.AppendLine("// <auto-generated/>");
            source.AppendLine("#nullable enable");
            string? ns = type.ContainingNamespace.IsGlobalNamespace ? null : type.ContainingNamespace.ToDisplayString();
            if (ns is not null)
                source.Append("namespace ").Append(ns).AppendLine(" {");
            source.Append("    partial record ").Append(self).AppendLine(" {");

            // To the generic form.
            source.AppendLine("        /// <summary>");
            source.Append("        ///     The generic form (DR-13): ").AppendLine(asDict
                ? "an <c>Info</c> with the members by Python name, in declaration order, as a <c>Dict</c> space samples it."
                : "the members as an <see langword=\"object\"/>?[], Gym.NET's Python tuple, as a <c>Tuple</c> space samples it.");
            source.AppendLine("        /// </summary>");
            source.AppendLine("        /// <param name=\"value\">The record.</param>");
            source.AppendLine("        /// <returns>The generic form; its values are the record's, not copies.</returns>");
            source.AppendLine("        /// <exception cref=\"global::System.ArgumentNullException\">Thrown when <paramref name=\"value\"/> is <see langword=\"null\"/>.</exception>");
            source.Append("        public static implicit operator ").Append(generic).Append('(').Append(self).AppendLine(" value) {");
            source.AppendLine("            if (value is null)");
            source.AppendLine("                throw new global::System.ArgumentNullException(nameof(value));");
            if (asDict) {
                source.AppendLine("            return new global::Gym.Envs.Info {");
                foreach (IParameterSymbol item in items)
                    source.Append("                [").Append(SymbolDisplay.FormatLiteral(PythonName(item, PropertyOf(type, item)), true)).Append("] = value.").Append(item.Name).AppendLine(",");
                source.AppendLine("            };");
            } else {
                source.Append("            return new object?[] { ").Append(string.Join(", ", items.Select(i => "value." + i.Name))).AppendLine(" };");
            }

            source.AppendLine("        }");
            source.AppendLine();

            // From the generic form.
            source.AppendLine("        /// <summary>The record of a generic form: the items in order (a tuple) or the entries by Python name (a dict), each cast to its member's type.</summary>");
            source.AppendLine("        /// <param name=\"value\">The generic form.</param>");
            source.AppendLine("        /// <returns>The record, holding the generic form's values.</returns>");
            source.AppendLine("        /// <exception cref=\"global::System.ArgumentNullException\">Thrown when <paramref name=\"value\"/> is <see langword=\"null\"/>.</exception>");
            source.AppendLine(asDict
                ? "        /// <exception cref=\"global::NumSharp.KeyError\">Thrown when a member's key is missing.</exception>"
                : "        /// <exception cref=\"global::NumSharp.ValueError\">Thrown when the item count is not the record's: <c>expected 3 items, got 2</c>.</exception>");
            source.AppendLine("        /// <exception cref=\"global::System.InvalidCastException\">Thrown when a value is not of its member's type.</exception>");
            source.Append("        public static implicit operator ").Append(self).Append('(').Append(generic).AppendLine(" value) {");
            source.AppendLine("            if (value is null)");
            source.AppendLine("                throw new global::System.ArgumentNullException(nameof(value));");
            if (!asDict) {
                source.Append("            if (value.Length != ").Append(items.Length.ToString(CultureInfo.InvariantCulture)).AppendLine(")");
                source.Append("                throw new global::NumSharp.ValueError(\"expected ").Append(items.Length.ToString(CultureInfo.InvariantCulture))
                    .AppendLine(" items, got \" + value.Length);");
            }

            source.Append("            return new ").Append(self).Append('(');
            source.Append(string.Join(", ", items.Select((item, index) => {
                string cast = "(" + item.Type.ToDisplayString(TypeFormat) + ")";
                string read = asDict ? "value[" + SymbolDisplay.FormatLiteral(PythonName(item, PropertyOf(type, item)), true) + "]" : "value[" + index.ToString(CultureInfo.InvariantCulture) + "]";
                return cast + read + "!";
            })));
            source.AppendLine(");");
            source.AppendLine("        }");
            source.AppendLine();

            // Python's == and a consistent hash.
            source.AppendLine("        /// <summary>Python's <c>==</c>: the generic forms are equal (<c>PyEquality</c>), member by member; arrays compare by value.</summary>");
            source.AppendLine("        /// <param name=\"other\">The other record.</param>");
            source.AppendLine("        /// <returns><see langword=\"true\"/> when the records are equal.</returns>");
            source.Append("        public ").Append(isSealed ? "" : "virtual ").Append("bool Equals(").Append(self).AppendLine("? other) =>");
            source.Append("            other is not null && global::Gym.Utils.PyEquality.AreEqual((").Append(generic).Append(")this, (").Append(generic).AppendLine(")other);");
            source.AppendLine();
            source.AppendLine("        /// <summary>A hash consistent with <see cref=\"Equals(object)\"/>: the member count, as array members have no value hash.</summary>");
            source.AppendLine("        /// <returns>The hash.</returns>");
            source.Append("        public override int GetHashCode() => ").Append(items.Length.ToString(CultureInfo.InvariantCulture)).AppendLine(";");
            source.AppendLine();
            source.Append("        /// <summary>Python's <c>repr</c> of the generic form: ").Append(asDict ? "<c>{'a': 1, 'b': 2}</c>" : "<c>(1, 2)</c>").AppendLine(".</summary>");
            source.AppendLine("        /// <returns>The text.</returns>");
            source.Append("        public override string ToString() => global::Gym.Utils.PyBuiltins.Repr((").Append(generic).AppendLine(")this);");
            source.AppendLine("    }");
            if (ns is not null)
                source.AppendLine("}");
            return new GeneratedFile($"{HintPrefix(type)}.{(asDict ? "Dict" : "Tuple")}.g.cs", source.ToString());
        }

        /// <summary>The property a positional record declares for a primary constructor parameter, where its attributes may sit.</summary>
        /// <param name="type">The record.</param>
        /// <param name="parameter">The parameter.</param>
        /// <returns>The property, or <see langword="null"/>.</returns>
        private static IPropertySymbol? PropertyOf(INamedTypeSymbol type, IParameterSymbol parameter) =>
            type.GetMembers(parameter.Name).OfType<IPropertySymbol>().FirstOrDefault();

        /// <summary>
        ///     A parameter's Python name, as <c>PyKwargs.ParameterName</c> computes it at run time: its <c>[PyName]</c> (or its
        ///     property's), else the snake_case of its name.
        /// </summary>
        /// <param name="parameter">The parameter.</param>
        /// <param name="property">The record property it declares, or <see langword="null"/>.</param>
        /// <returns>The Python name.</returns>
        private static string PythonName(IParameterSymbol parameter, IPropertySymbol? property = null) {
            foreach (AttributeData attribute in parameter.GetAttributes().Concat(property?.GetAttributes() ?? Enumerable.Empty<AttributeData>())) {
                if (attribute.AttributeClass?.ToDisplayString() == PyNameAttribute && attribute.ConstructorArguments.Length == 1
                    && attribute.ConstructorArguments[0].Value is string name)
                    return name;
            }

            return SnakeCase(parameter.Name);
        }

        /// <summary>
        ///     Gym.NET's snake_case rule (<c>PyAttributes.SnakeCase</c>), which must agree with the run-time binding: a word
        ///     starts at an upper-case letter after a lower-case one or a digit, and at the last capital of an acronym.
        /// </summary>
        /// <param name="name">A C# name.</param>
        /// <returns>The snake_case name.</returns>
        private static string SnakeCase(string name) {
            var text = new StringBuilder(name.Length + 4);
            for (int i = 0; i < name.Length; i++) {
                char c = name[i];
                if (char.IsUpper(c)) {
                    bool afterLower = i > 0 && (char.IsLower(name[i - 1]) || char.IsDigit(name[i - 1]));
                    bool endsAcronym = i > 0 && char.IsUpper(name[i - 1]) && i + 1 < name.Length && char.IsLower(name[i + 1]);
                    if (afterLower || endsAcronym)
                        text.Append('_');
                    text.Append(char.ToLowerInvariant(c));
                } else {
                    text.Append(c);
                }
            }

            return text.ToString();
        }

        /// <summary>A parameter name as a property name: PascalCase, the words of a snake_case name joined (<c>enable_wind</c> is <c>EnableWind</c>).</summary>
        /// <param name="name">The parameter name.</param>
        /// <returns>The property name.</returns>
        private static string PropertyName(string name) {
            string[] words = name.Split(new[] { '_' }, StringSplitOptions.RemoveEmptyEntries);
            return string.Concat(words.Select(word => char.ToUpperInvariant(word[0]) + word.Substring(1)));
        }

        /// <summary>
        ///     A constructor parameter's default as a C# expression of its type: a literal with the type's suffix, an enum
        ///     member's cast, <c>default!</c> for <see langword="null"/>.
        /// </summary>
        /// <param name="value">The default value Roslyn reports.</param>
        /// <param name="type">The parameter's type.</param>
        /// <returns>The expression.</returns>
        private static string Literal(object? value, ITypeSymbol type) {
            if (value is null)
                return "default!";
            ITypeSymbol target = type is INamedTypeSymbol { OriginalDefinition.SpecialType: SpecialType.System_Nullable_T } nullable ? nullable.TypeArguments[0] : type;
            string typeName = target.ToDisplayString(TypeFormat);
            if (target.TypeKind == TypeKind.Enum)
                return "(" + typeName + ")(" + SymbolDisplay.FormatPrimitive(value, quoteStrings: true, useHexadecimalNumbers: false) + ")";
            switch (value) {
                case bool flag:
                    return flag ? "true" : "false";
                case string text:
                    return SymbolDisplay.FormatLiteral(text, quote: true);
                case char character:
                    return SymbolDisplay.FormatLiteral(character, quote: true);
                case float single:
                    return float.IsNaN(single) ? "float.NaN" : float.IsPositiveInfinity(single) ? "float.PositiveInfinity"
                        : float.IsNegativeInfinity(single) ? "float.NegativeInfinity" : single.ToString("R", CultureInfo.InvariantCulture) + "F";
                case double number:
                    return double.IsNaN(number) ? "double.NaN" : double.IsPositiveInfinity(number) ? "double.PositiveInfinity"
                        : double.IsNegativeInfinity(number) ? "double.NegativeInfinity" : number.ToString("R", CultureInfo.InvariantCulture) + "D";
                case decimal money:
                    return money.ToString(CultureInfo.InvariantCulture) + "M";
                default:
                    // Every integer type: a cast keeps narrow types exact, the literal itself is in range.
                    return "(" + typeName + ")" + SymbolDisplay.FormatPrimitive(value, quoteStrings: true, useHexadecimalNumbers: false);
            }
        }

        /// <summary>
        ///     The <c>&lt;param&gt;</c> texts of a constructor's documentation, by parameter name, as single lines. Read from the
        ///     compiler's documentation when the project produces it, else from the <c>///</c> comment lines themselves, which
        ///     the compiler keeps as plain trivia when documentation is off.
        /// </summary>
        /// <param name="constructor">The constructor.</param>
        /// <returns>The texts.</returns>
        private static Dictionary<string, string> ParameterDocs(IMethodSymbol constructor) {
            var docs = new Dictionary<string, string>(StringComparer.Ordinal);
            string? xml = constructor.GetDocumentationCommentXml();
            if (string.IsNullOrWhiteSpace(xml))
                xml = DocumentationFromTrivia(constructor);
            if (string.IsNullOrWhiteSpace(xml))
                return docs;
            try {
                foreach (XElement param in XElement.Parse(xml).Elements("param")) {
                    string? name = param.Attribute("name")?.Value;
                    if (name is null)
                        continue;
                    // The inner XML, so <c> and <see> survive, collapsed to one line for the generated comment.
                    string inner = string.Concat(param.Nodes().Select(node => node.ToString()));
                    docs[name] = string.Join(" ", inner.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries).Select(line => line.Trim()));
                }
            } catch (System.Xml.XmlException) {
                // Malformed documentation: the generic text stands in.
            }

            return docs;
        }

        /// <summary>
        ///     A declaration's <c>///</c> comment as one XML document, from its leading trivia: the lines without their
        ///     <c>///</c>, wrapped in a root element, as the compiler would have read them.
        /// </summary>
        /// <param name="symbol">The documented symbol.</param>
        /// <returns>The XML, or <see langword="null"/> when the declaration has no <c>///</c> lines.</returns>
        private static string? DocumentationFromTrivia(ISymbol symbol) {
            SyntaxReference? reference = symbol.DeclaringSyntaxReferences.FirstOrDefault();
            if (reference is null)
                return null;
            var lines = new List<string>();
            foreach (SyntaxTrivia trivia in reference.GetSyntax().GetLeadingTrivia()) {
                // Structured trivia when documentation is on, plain single-line comments when it is off: both read as text.
                foreach (string line in trivia.ToFullString().Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)) {
                    string trimmed = line.TrimStart();
                    if (trimmed.StartsWith("///", StringComparison.Ordinal))
                        lines.Add(trimmed.Substring(3));
                }
            }

            return lines.Count == 0 ? null : "<doc>" + string.Join("\n", lines) + "</doc>";
        }

        /// <summary>Whether a type mentions a type parameter anywhere (as itself, an array element, or a type argument).</summary>
        /// <param name="type">The type.</param>
        /// <param name="parameter">The type parameter.</param>
        /// <returns><see langword="true"/> when it does.</returns>
        private static bool Mentions(ITypeSymbol type, ITypeParameterSymbol parameter) => type switch {
            ITypeParameterSymbol candidate => SymbolEqualityComparer.Default.Equals(candidate, parameter),
            IArrayTypeSymbol array => Mentions(array.ElementType, parameter),
            INamedTypeSymbol named => named.TypeArguments.Any(argument => Mentions(argument, parameter)),
            _ => false,
        };

        /// <summary>The <c>cref</c> of a type in generated documentation: its name, with <c>{T}</c> for type parameters.</summary>
        /// <param name="type">The type.</param>
        /// <returns>The cref text.</returns>
        private static string DocCref(INamedTypeSymbol type) =>
            type.TypeParameters.Length == 0 ? type.Name : type.Name + "{" + string.Join(", ", type.TypeParameters.Select(t => t.Name)) + "}";

        /// <summary>A unique, file-system-safe prefix for a type's generated files: its full metadata name.</summary>
        /// <param name="type">The type.</param>
        /// <returns>The prefix.</returns>
        private static string HintPrefix(INamedTypeSymbol type) =>
            (type.ContainingNamespace.IsGlobalNamespace ? "" : type.ContainingNamespace.ToDisplayString() + ".") + type.MetadataName.Replace('`', '_');

        /// <summary>One generated file, compared by value so the incremental pipeline can cache it.</summary>
        private sealed class GeneratedFile : IEquatable<GeneratedFile> {
            /// <summary>Holds the file.</summary>
            /// <param name="hintName">The file's name in the compilation.</param>
            /// <param name="source">The C# text.</param>
            public GeneratedFile(string hintName, string source) {
                HintName = hintName;
                Source = source;
            }

            /// <summary>The file's name in the compilation.</summary>
            public string HintName { get; }

            /// <summary>The C# text.</summary>
            public string Source { get; }

            /// <summary>Equal when both the name and the text are.</summary>
            /// <param name="other">The other file.</param>
            /// <returns><see langword="true"/> when equal.</returns>
            public bool Equals(GeneratedFile? other) => other is not null && HintName == other.HintName && Source == other.Source;

            /// <summary>Equal when <paramref name="obj"/> is an equal file.</summary>
            /// <param name="obj">The other object.</param>
            /// <returns><see langword="true"/> when equal.</returns>
            public override bool Equals(object? obj) => Equals(obj as GeneratedFile);

            /// <summary>A hash of the name and the text.</summary>
            /// <returns>The hash.</returns>
            public override int GetHashCode() => (HintName.GetHashCode() * 397) ^ Source.GetHashCode();
        }
    }
}
