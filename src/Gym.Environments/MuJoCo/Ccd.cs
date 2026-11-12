using System;

namespace Gym.MuJoCo {
    /// <summary>Which support function a <see cref="CcdObject"/> answers with.</summary>
    internal enum CcdSupport {
        /// <summary>The geom's own shape (sphere, capsule or cylinder).</summary>
        Shape,

        /// <summary>A capsule's segment (<c>mjc_lineSupport</c>), while MuJoCo collides it as a line.</summary>
        Line,

        /// <summary>A sphere's centre (<c>mjc_pointSupport</c>), while MuJoCo collides it as a point.</summary>
        Point,
    }

    /// <summary>
    ///     A geom as MuJoCo's convex collider sees it (<c>mjCCDObj</c>): its type, size, frame, the pair's margin, the support
    ///     function in use, and the support vertices GJK and EPA leave on it for the multi-contact pass. The collider swaps the
    ///     support (sphere and capsule shrinking) and zeroes the margin in place, and restores both itself.
    /// </summary>
    internal sealed class CcdObject {
        /// <summary>The geom type (2 sphere, 3 capsule, 5 cylinder).</summary>
        public int GeomType;
        /// <summary>The geom size (sphere and capsule: radius, half-length; cylinder: radius, half-height).</summary>
        public readonly double[] Size = new double[3];
        /// <summary>The geom position.</summary>
        public readonly double[] Pos = new double[3];
        /// <summary>The geom rotation matrix.</summary>
        public readonly double[] Mat = new double[9];
        /// <summary>The pair's margin; half of it inflates each support point while it is nonzero.</summary>
        public double Margin;
        /// <summary>The support function in use.</summary>
        public CcdSupport Support;
        /// <summary>
        ///     The last support point's vertex index (<c>vertindex</c>): the cylinder's support writes 0 for its top disk and 1
        ///     for its bottom one; the other supports leave it at -1.
        /// </summary>
        public int VertIndex;
        /// <summary>The number of support vertices GJK or EPA left (<c>nsupp</c>), 0 to 4.</summary>
        public int NSupp;
        /// <summary>The support vertices on this geom, 3 values each (<c>suppvert</c>).</summary>
        public readonly double[] SuppVert = new double[12];
        /// <summary>Their vertex indices (<c>suppind</c>).</summary>
        public readonly int[] SuppInd = new int[4];

        /// <summary>Sets the object from a geom (<c>mjc_initCCDObj</c>).</summary>
        /// <param name="type">The geom type.</param>
        /// <param name="size">The geom size.</param>
        /// <param name="pos">The geom position.</param>
        /// <param name="mat">The geom rotation matrix.</param>
        /// <param name="margin">The pair's margin.</param>
        public void Init(int type, ReadOnlySpan<double> size, ReadOnlySpan<double> pos, ReadOnlySpan<double> mat, double margin) {
            GeomType = type;
            size.Slice(0, 3).CopyTo(Size);
            pos.Slice(0, 3).CopyTo(Pos);
            mat.Slice(0, 9).CopyTo(Mat);
            Margin = margin;
            Support = CcdSupport.Shape;
            VertIndex = -1;
            NSupp = 0;
        }

        /// <summary>
        ///     The support point in a unit direction: the shape's farthest point along it (<c>mjc_*Support</c>), each in
        ///     MuJoCo's arithmetic. The cylinder's records which disk it took in <see cref="VertIndex"/>.
        /// </summary>
        /// <param name="res">Receives the point.</param>
        /// <param name="dir">The direction (normalized by the caller).</param>
        /// <exception cref="NotSupportedException">A geom type the collider does not reproduce.</exception>
        public void SupportPoint(Span<double> res, ReadOnlySpan<double> dir) {
            if (Support == CcdSupport.Point) {
                res[0] = Pos[0];
                res[1] = Pos[1];
                res[2] = Pos[2];
                return;
            }
            if (Support == CcdSupport.Line) {
                double dot = Mat[2] * dir[0] + Mat[5] * dir[1] + Mat[8] * dir[2];
                double scl = dot >= 0 ? Size[1] : -Size[1];
                res[0] = Mat[2] * scl + Pos[0];
                res[1] = Mat[5] * scl + Pos[1];
                res[2] = Mat[8] * scl + Pos[2];
                return;
            }
            Span<double> local = stackalloc double[3], supp = stackalloc double[3];
            switch (GeomType) {
                case 2:
                    res[0] = Size[0] * dir[0] + Pos[0];
                    res[1] = Size[0] * dir[1] + Pos[1];
                    res[2] = Size[0] * dir[2] + Pos[2];
                    return;
                case 3:
                    Spatial.MulMatTVec3(local, Mat, dir);
                    supp[0] = local[0] * Size[0];
                    supp[1] = local[1] * Size[0];
                    supp[2] = local[2] * Size[0];
                    supp[2] += local[2] >= 0 ? Size[1] : -Size[1];
                    break;
                case 5:
                    Spatial.MulMatTVec3(local, Mat, dir);
                    double n2 = local[0] * local[0] + local[1] * local[1];
                    double s = n2 >= Ccd.MinVal2 ? Size[0] / Math.Sqrt(n2) : 0;
                    supp[0] = s * local[0];
                    supp[1] = s * local[1];
                    supp[2] = local[2] >= 0 ? Size[1] : -Size[1];
                    VertIndex = local[2] >= 0 ? 0 : 1;
                    break;
                default:
                    throw new NotSupportedException($"the convex collider: geom type {GeomType}");
            }
            // localToGlobal: the rotation first, then the position
            Spatial.MulMatVec3(res, Mat, supp);
            res[0] += Pos[0];
            res[1] += Pos[1];
            res[2] += Pos[2];
        }
    }

    /// <summary>
    ///     The state of one convex collision query (<c>mjCCDStatus</c>): the witness points and distances it found, its
    ///     configuration and statistics.
    /// </summary>
    internal sealed class CcdStatus {
        /// <summary>Whether GJK proved the geoms apart (or found a separating direction).</summary>
        public bool Separated;
        /// <summary>The distance at each witness pair (negative: penetration).</summary>
        public readonly double[] Dist = new double[Ccd.MaxConPair];
        /// <summary>The witness points on geom 1 (3 per pair).</summary>
        public readonly double[] X1 = new double[3 * Ccd.MaxConPair];
        /// <summary>The witness points on geom 2.</summary>
        public readonly double[] X2 = new double[3 * Ccd.MaxConPair];
        /// <summary>The number of witness pairs.</summary>
        public int Nx;
        /// <summary>The iteration budget of GJK and EPA.</summary>
        public int MaxIterations;
        /// <summary>The tolerance of GJK and EPA.</summary>
        public double Tolerance;
        /// <summary>The number of contacts wanted, capped at <see cref="Ccd.MaxConPair"/>.</summary>
        public int MaxContacts;
        /// <summary>The distance beyond which GJK may stop proving separation.</summary>
        public double DistCutoff;
        /// <summary>GJK's iterations.</summary>
        public int GjkIterations;
        /// <summary>EPA's iterations.</summary>
        public int EpaIterations;
        /// <summary>EPA's outcome (<c>mjEPAStatus</c>; <see cref="Ccd.EpaNoContact"/> when EPA did not run).</summary>
        public int EpaStatus;
    }

    /// <summary>
    ///     MuJoCo 4.0.0's native convex collision (<c>mjc_ccd</c> of <c>engine_collision_gjk.c</c>): GJK with the
    ///     signed-volume subalgorithm for distances, EPA on an expanding polytope for the penetration depth, and the
    ///     multi-contact pass that clips a capsule's edge against a cylinder's face, written to MuJoCo's arithmetic order.
    ///     Pusher reaches it for the gripper's capsules against the object's cylinder.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Implemented as far as the bundled models reach: spheres and capsules shrink to points and lines and margins
    ///         are taken out of GJK (shallow contacts are inflated back), deep contacts run GJK on the full shapes and EPA
    ///         from GJK's segment, triangle or tetrahedron, and a capsule against a cylinder asks for two contacts, so a
    ///         penetrating pair goes through <see cref="Multicontact"/>. The discrete-geom paths (meshes, boxes) and the
    ///         face-face and plane cases of the multi-contact pass throw <see cref="NotSupportedException"/> by name.
    ///     </para>
    ///     <para>
    ///         Reference: MuJoCo 4.0.0's
    ///         <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_collision_gjk.c">engine_collision_gjk.c</see>
    ///         and <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_collision_convex.c">engine_collision_convex.c</see>
    ///         (Apache-2.0).
    ///     </para>
    /// </remarks>
    internal static class Ccd {
        /// <summary>Doubles per polytope vertex: the difference point, the point on geom 1, the point on geom 2.</summary>
        public const int VertexSize = 9;
        /// <summary>MuJoCo's <c>mjMAXCONPAIR</c>, the witness capacity.</summary>
        public const int MaxConPair = 50;
        /// <summary><c>mjMINVAL*mjMINVAL</c>.</summary>
        public const double MinVal2 = Kernels.MinVal * Kernels.MinVal;
        /// <summary><c>mjMAX_LIMIT</c> (<c>DBL_MAX</c>): "no distance".</summary>
        public const double MaxLimit = double.MaxValue;
        /// <summary><c>mjMINEPATOL</c> in double precision.</summary>
        private const double MinEpaTol = Kernels.MinVal;
        /// <summary><c>mjMINEPATOL2</c>.</summary>
        private const double MinEpaTol2 = MinEpaTol * MinEpaTol;
        /// <summary><c>mjMINDIST2</c>, <c>mjMINDIST3</c> and <c>mjMINDIST4</c> in double precision: a face this close to the origin holds it.</summary>
        private const double MinDist = MinVal2;
        /// <summary><c>mjFACE_TOL</c>: two face normals this anti-parallel (cosine) are aligned.</summary>
        private const double FaceTol = 0.996;
        /// <summary><c>mjEDGE_TOL</c>: an edge this perpendicular (cosine) to a face normal lies in the face.</summary>
        private const double EdgeTol = 0.0888;
        /// <summary>EPA status: no contact (<c>mjEPA_NOCONTACT</c>).</summary>
        public const int EpaNoContact = -1;
        /// <summary>EPA status: the hexahedron from a segment is not convex (<c>mjEPA_P2_NONCONVEX</c>).</summary>
        private const int P2Nonconvex = 2;
        /// <summary>EPA status: the triangle's normal is degenerate (<c>mjEPA_P3_BAD_NORMAL</c>).</summary>
        private const int P3BadNormal = 4;
        /// <summary>EPA status: the support point along the normal lies in the triangle (<c>mjEPA_P3_INVALID_V4</c>).</summary>
        private const int P3InvalidV4 = 5;
        /// <summary>EPA status: the support point against the normal lies in the triangle (<c>mjEPA_P3_INVALID_V5</c>).</summary>
        private const int P3InvalidV5 = 6;
        /// <summary>EPA status: the hexahedron misses the origin (<c>mjEPA_P3_MISSING_ORIGIN</c>).</summary>
        private const int P3MissingOrigin = 7;
        /// <summary>EPA status: the origin lies on a face of the hexahedron (<c>mjEPA_P3_ORIGIN_ON_FACE</c>).</summary>
        private const int P3OriginOnFace = 8;
        /// <summary>EPA status when the origin is not in the initial tetrahedron (<c>mjEPA_P4_MISSING_ORIGIN</c>).</summary>
        private const int P4MissingOrigin = 9;

        // ------------------------------------------------------------------------------------------ the query

        /// <summary>
        ///     One convex query (<c>mjc_ccd</c>) with no distance cutoff, as <c>mjc_penetration</c> configures it: GJK from the
        ///     geom centres with spheres and capsules shrunk and the margins taken out; a shallow contact is inflated back and,
        ///     when it penetrates and more than one contact is wanted, handed to <see cref="Multicontact"/>; a deep one runs GJK
        ///     on the full shapes, EPA and the multi-contact pass.
        /// </summary>
        /// <param name="status">Receives the witness points, distances and statistics.</param>
        /// <param name="poly">The polytope workspace, sized for <paramref name="maxIterations"/>.</param>
        /// <param name="obj1">The first geom (its support and margin are restored on return).</param>
        /// <param name="obj2">The second geom (likewise).</param>
        /// <param name="maxIterations">GJK's and EPA's budget (<c>opt.ccd_iterations</c>).</param>
        /// <param name="tolerance">Their tolerance (<c>opt.ccd_tolerance</c>).</param>
        /// <param name="maxContacts">The contacts wanted (<c>maxContacts</c>: 2 for a capsule against a cylinder).</param>
        /// <returns>The smallest witness distance (negative: penetration), or <see cref="MaxLimit"/>.</returns>
        /// <exception cref="NotSupportedException">A multi-contact case beyond a capsule's edge against a cylinder's face.</exception>
        public static double Run(CcdStatus status, CcdPolytope poly, CcdObject obj1, CcdObject obj2, int maxIterations, double tolerance,
                                 int maxContacts) {
            const double configCutoff = 0;
            // mjc_center: a geom's position
            obj1.Pos.CopyTo(status.X1, 0);
            obj2.Pos.CopyTo(status.X2, 0);
            status.GjkIterations = 0;
            status.EpaIterations = 0;
            status.EpaStatus = EpaNoContact;
            status.Tolerance = tolerance;
            status.MaxIterations = maxIterations;
            status.MaxContacts = maxContacts < MaxConPair ? maxContacts : MaxConPair;
            status.DistCutoff = configCutoff;

            // spheres and capsules shrink to points and lines, and both geoms' margins leave GJK (both are geoms here)
            if (obj1.GeomType == 2 || obj2.GeomType == 2 || obj1.GeomType == 3 || obj2.GeomType == 3 || obj1.Margin > 0 || obj2.Margin > 0) {
                CcdSupport support1 = obj1.Support, support2 = obj2.Support;
                double margin1 = obj1.Margin, margin2 = obj2.Margin;
                double fullMargin1 = 0.5 * margin1;
                obj1.Margin = 0;
                if (obj1.GeomType == 2) {
                    fullMargin1 += obj1.Size[0];
                    obj1.Support = CcdSupport.Point;
                } else if (obj1.GeomType == 3) {
                    fullMargin1 += obj1.Size[0];
                    obj1.Support = CcdSupport.Line;
                }
                double fullMargin2 = 0.5 * margin2;
                obj2.Margin = 0;
                if (obj2.GeomType == 2) {
                    fullMargin2 += obj2.Size[0];
                    obj2.Support = CcdSupport.Point;
                } else if (obj2.GeomType == 3) {
                    fullMargin2 += obj2.Size[0];
                    obj2.Support = CcdSupport.Line;
                }
                status.DistCutoff += fullMargin1 + fullMargin2;
                Gjk(status, obj1, obj2);
                status.DistCutoff = configCutoff;
                obj1.Margin = margin1;
                obj2.Margin = margin2;
                obj1.Support = support1;
                obj2.Support = support2;
                // shallow penetration: inflate the contact back to the shapes
                if (status.Dist[0] > status.Tolerance) {
                    Inflate(status.Nx, status.X1, status.X2, status.Dist, fullMargin1, fullMargin2, null);
                    if (status.Dist[0] > status.DistCutoff) {
                        status.Dist[0] = MaxLimit;
                    } else if (maxContacts > 1 && status.Dist[0] < 0 && obj1.NSupp > 0) {
                        status.Nx = Multicontact(obj1, obj2, status.X1, status.X2, status.Dist, status.MaxContacts);
                        return MinDistance(status);
                    }
                    return status.Dist[0];
                }
                if (maxContacts == 0) {
                    status.Nx = 0;
                    status.Dist[0] = 0;
                    return 0;
                }
                // deep penetration: GJK again on the full shapes, from the centres
                status.GjkIterations = 0;
                obj1.Pos.CopyTo(status.X1, 0);
                obj2.Pos.CopyTo(status.X2, 0);
            }
            Gjk(status, obj1, obj2);
            if (maxContacts == 0)
                return status.Dist[0];
            if (status.Dist[0] <= tolerance && obj1.NSupp > 1 && !status.Separated) {
                double gjkDist = status.Dist[0];
                status.Dist[0] = 0;
                poly.Reset(maxIterations);
                int ret = obj1.NSupp == 2 ? Polytope2(poly, status, obj1, obj2)
                        : obj1.NSupp == 3 ? Polytope3(poly, status, obj1, obj2, true)
                        : Polytope4(poly, status, obj1, obj2);
                status.EpaStatus = ret;
                if (ret == 0) {
                    int face = Epa(status, poly, obj1, obj2);
                    if (maxContacts > 1 && face >= 0)
                        status.Nx = Multicontact(obj1, obj2, status.X1, status.X2, status.Dist, status.MaxContacts);
                    // a distance query that found no contact gets GJK's distance back (no cutoff here, so never)
                    if (face < 0 && configCutoff > 0 && status.EpaStatus == EpaNoContact) {
                        status.Nx = 1;
                        status.Dist[0] = gjkDist;
                    }
                }
            }
            return MinDistance(status);
        }

        /// <summary>The smallest of the status's witness distances, the first's on ties.</summary>
        /// <param name="status">The status.</param>
        /// <returns>The distance.</returns>
        private static double MinDistance(CcdStatus status) {
            double minDist = status.Dist[0];
            for (int i = 1; i < status.Nx; i++)
                if (status.Dist[i] < minDist)
                    minDist = status.Dist[i];
            return minDist;
        }

        /// <summary>
        ///     Moves witness pairs outward by the margins a shrunk geom or a removed margin left out (<c>inflate</c>): along
        ///     <paramref name="normal"/> when given, else along the first pair's direction from geom 1 to geom 2; each distance
        ///     loses both margins.
        /// </summary>
        /// <param name="nx">The witness pairs.</param>
        /// <param name="x1">The points on geom 1, moved by <paramref name="margin1"/>.</param>
        /// <param name="x2">The points on geom 2, moved back by <paramref name="margin2"/>.</param>
        /// <param name="dist">The distances, or <see langword="null"/>.</param>
        /// <param name="margin1">Geom 1's margin left out.</param>
        /// <param name="margin2">Geom 2's margin left out.</param>
        /// <param name="normal">The direction, or <see langword="null"/> for the first pair's.</param>
        private static void Inflate(int nx, double[] x1, double[] x2, double[]? dist, double margin1, double margin2, double[]? normal) {
            if (margin1 == 0 && margin2 == 0)
                return;
            Span<double> n = stackalloc double[3];
            if (normal != null) {
                n[0] = normal[0];
                n[1] = normal[1];
                n[2] = normal[2];
            } else {
                n[0] = x2[0] - x1[0];
                n[1] = x2[1] - x1[1];
                n[2] = x2[2] - x1[2];
                Spatial.Normalize3(n);
            }
            for (int i = 0; i < nx; i++) {
                if (margin1 != 0)
                    for (int k = 0; k < 3; k++)
                        x1[3 * i + k] = x1[3 * i + k] + margin1 * n[k];
                if (margin2 != 0)
                    for (int k = 0; k < 3; k++)
                        x2[3 * i + k] = x2[3 * i + k] + -margin2 * n[k];
                if (dist != null)
                    dist[i] -= margin1 + margin2;
            }
        }

        // ------------------------------------------------------------------------------------------ vector helpers

        /// <summary><c>dot3</c> of two 3-vectors, left to right.</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns>The sum of the three products.</returns>
        private static double Dot3(ReadOnlySpan<double> a, ReadOnlySpan<double> b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

        /// <summary><c>norm3</c>: the square root of <see cref="Dot3"/> of a vector with itself.</summary>
        /// <param name="v">The vector.</param>
        /// <returns>Its length.</returns>
        private static double Norm3(ReadOnlySpan<double> v) => Math.Sqrt(Dot3(v, v));

        /// <summary>
        ///     <c>det3</c>: the determinant of the matrix with columns <paramref name="v1"/>, <paramref name="v2"/>,
        ///     <paramref name="v3"/>, as <c>v1 · (v2 x v3)</c> written out.
        /// </summary>
        /// <param name="v1">The first column.</param>
        /// <param name="v2">The second column.</param>
        /// <param name="v3">The third column.</param>
        /// <returns>The determinant, positive for a right-handed triple.</returns>
        private static double Det3(ReadOnlySpan<double> v1, ReadOnlySpan<double> v2, ReadOnlySpan<double> v3) =>
            v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) + v1[1] * (v2[2] * v3[0] - v2[0] * v3[2]) + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);

        /// <summary><c>sameSign2</c>: 1 if both are positive, -1 if both negative, else 0.</summary>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        /// <returns>1, -1 or 0; a zero gives 0, which the subalgorithms read as disagreement.</returns>
        private static int SameSign2(double a, double b) => a > 0 && b > 0 ? 1 : a < 0 && b < 0 ? -1 : 0;

        /// <summary>
        ///     <c>lincomb</c>: <c>sum coef[i]*v_i</c> over the first <paramref name="n"/> rows of an array, each component
        ///     summed left to right.
        /// </summary>
        /// <param name="res">Receives the combination.</param>
        /// <param name="coef">The coefficients.</param>
        /// <param name="n">How many (0 to 4).</param>
        /// <param name="v">The vectors' array.</param>
        /// <param name="o1">The first vector's offset.</param>
        /// <param name="o2">The second's.</param>
        /// <param name="o3">The third's.</param>
        /// <param name="o4">The fourth's.</param>
        private static void Lincomb(Span<double> res, ReadOnlySpan<double> coef, int n, ReadOnlySpan<double> v, int o1, int o2, int o3, int o4) {
            switch (n) {
                case 0:
                    res[0] = res[1] = res[2] = 0;
                    break;
                case 1:
                    for (int k = 0; k < 3; k++)
                        res[k] = coef[0] * v[o1 + k];
                    break;
                case 2:
                    for (int k = 0; k < 3; k++)
                        res[k] = coef[0] * v[o1 + k] + coef[1] * v[o2 + k];
                    break;
                case 3:
                    for (int k = 0; k < 3; k++)
                        res[k] = coef[0] * v[o1 + k] + coef[1] * v[o2 + k] + coef[2] * v[o3 + k];
                    break;
                case 4:
                    for (int k = 0; k < 3; k++)
                        res[k] = coef[0] * v[o1 + k] + coef[1] * v[o2 + k] + coef[2] * v[o3 + k] + coef[3] * v[o4 + k];
                    break;
            }
        }

        /// <summary>
        ///     The origin projected on a triangle's plane (<c>projectOriginPlane</c>), trying the normals at the second, first
        ///     and third vertex in turn until one is usable.
        /// </summary>
        /// <param name="res">Receives the projection; untouched for a degenerate triangle.</param>
        /// <param name="v1">The first vertex.</param>
        /// <param name="v2">The second vertex.</param>
        /// <param name="v3">The third vertex.</param>
        /// <returns>1 for a degenerate triangle (zero normal), else 0.</returns>
        private static int ProjectOriginPlane(Span<double> res, ReadOnlySpan<double> v1, ReadOnlySpan<double> v2, ReadOnlySpan<double> v3) {
            Span<double> d21 = stackalloc double[3], d31 = stackalloc double[3], d32 = stackalloc double[3], n = stackalloc double[3];
            for (int k = 0; k < 3; k++) {
                d21[k] = v2[k] - v1[k];
                d31[k] = v3[k] - v1[k];
                d32[k] = v3[k] - v2[k];
            }
            Spatial.Cross(n, d32, d21);
            double nv = Dot3(n, v2), nn = Dot3(n, n);
            if (nn == 0)
                return 1;
            if (nv != 0 && nn > Kernels.MinVal) {
                Scale(res, n, nv / nn);
                return 0;
            }
            Spatial.Cross(n, d21, d31);
            nv = Dot3(n, v1);
            nn = Dot3(n, n);
            if (nn == 0)
                return 1;
            if (nv != 0 && nn > Kernels.MinVal) {
                Scale(res, n, nv / nn);
                return 0;
            }
            Spatial.Cross(n, d31, d32);
            nv = Dot3(n, v3);
            nn = Dot3(n, n);
            if (nn == 0)
                return 1;
            Scale(res, n, nv / nn);
            return 0;
        }

        /// <summary>
        ///     The origin projected on a line (<c>projectOriginLine</c>): <c>v2 - (v2·d / d·d)*d</c> with <c>d = v2 - v1</c>.
        /// </summary>
        /// <param name="res">Receives the projection.</param>
        /// <param name="v1">The line's first point.</param>
        /// <param name="v2">Its second point.</param>
        /// <returns>1 when the points coincide within <c>mjMINVAL2</c>, else 0.</returns>
        private static int ProjectOriginLine(Span<double> res, ReadOnlySpan<double> v1, ReadOnlySpan<double> v2) {
            Span<double> diff = stackalloc double[3] { v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2] };
            double dd = Dot3(diff, diff);
            if (dd < MinVal2)
                return 1;
            double scl = -(Dot3(v2, diff) / dd);
            res[0] = v2[0] + scl * diff[0];
            res[1] = v2[1] + scl * diff[1];
            res[2] = v2[2] + scl * diff[2];
            return 0;
        }

        /// <summary><c>scl3</c>: <c>s*v</c>.</summary>
        /// <param name="res">Receives the scaled vector; may alias <paramref name="v"/>.</param>
        /// <param name="v">The vector.</param>
        /// <param name="s">The factor.</param>
        private static void Scale(Span<double> res, ReadOnlySpan<double> v, double s) {
            res[0] = s * v[0];
            res[1] = s * v[1];
            res[2] = s * v[2];
        }

        // ------------------------------------------------------------------------------------------ GJK

        /// <summary>
        ///     Whether a geom is discrete for GJK and EPA (<c>discreteGeom</c>): no margin and a point or segment support, or a
        ///     mesh, box or height field; none of this collider's shapes with a margin.
        /// </summary>
        /// <param name="obj">The geom.</param>
        /// <returns>Whether its support has finitely many points.</returns>
        private static bool DiscreteGeom(CcdObject obj) {
            if (obj.Margin != 0)
                return false;
            if (obj.Support == CcdSupport.Point || obj.Support == CcdSupport.Line)
                return true;
            return obj.GeomType == 7 || obj.GeomType == 6 || obj.GeomType == 1;
        }

        /// <summary>
        ///     The support vertex of the Minkowski difference along a unit direction (<c>support</c>): geom 1's support along
        ///     it, geom 2's against it, each pushed out by half its margin while the margin is set, and their difference; the
        ///     vertex indices are the supports' <see cref="CcdObject.VertIndex"/>.
        /// </summary>
        /// <param name="v">Receives the vertex: difference, point on geom 1, point on geom 2.</param>
        /// <param name="o">The vertex's offset in <paramref name="v"/>.</param>
        /// <param name="index1">Receives geom 1's vertex index.</param>
        /// <param name="index2">Receives geom 2's vertex index.</param>
        /// <param name="obj1">Geom 1.</param>
        /// <param name="obj2">Geom 2.</param>
        /// <param name="dir">The direction.</param>
        /// <param name="dirNeg">Its negation.</param>
        private static void Support(Span<double> v, int o, out int index1, out int index2, CcdObject obj1, CcdObject obj2,
                                    ReadOnlySpan<double> dir, ReadOnlySpan<double> dirNeg) {
            Span<double> v1 = v.Slice(o + 3, 3), v2 = v.Slice(o + 6, 3);
            obj1.SupportPoint(v1, dir);
            if (obj1.Margin > 0) {
                double margin = 0.5 * obj1.Margin;
                v1[0] += dir[0] * margin;
                v1[1] += dir[1] * margin;
                v1[2] += dir[2] * margin;
            }
            obj2.SupportPoint(v2, dirNeg);
            if (obj2.Margin > 0) {
                double margin = 0.5 * obj2.Margin;
                v2[0] += dirNeg[0] * margin;
                v2[1] += dirNeg[1] * margin;
                v2[2] += dirNeg[2] * margin;
            }
            v[o] = v1[0] - v2[0];
            v[o + 1] = v1[1] - v2[1];
            v[o + 2] = v1[2] - v2[2];
            index1 = obj1.VertIndex;
            index2 = obj2.VertIndex;
        }

        /// <summary>GJK's support vertex toward the origin from <paramref name="xk"/> (<c>gjkSupport</c>).</summary>
        /// <param name="v">Receives the vertex (9 values at <paramref name="o"/>).</param>
        /// <param name="o">The vertex's offset.</param>
        /// <param name="index1">Receives geom 1's vertex index.</param>
        /// <param name="index2">Receives geom 2's vertex index.</param>
        /// <param name="obj1">Geom 1, supported along <c>-xk</c>.</param>
        /// <param name="obj2">Geom 2, supported along <c>xk</c>.</param>
        /// <param name="xk">The current point of the Minkowski difference.</param>
        /// <param name="xNorm">Its norm, nonzero.</param>
        private static void GjkSupport(Span<double> v, int o, out int index1, out int index2, CcdObject obj1, CcdObject obj2,
                                       ReadOnlySpan<double> xk, double xNorm) {
            Span<double> dir = stackalloc double[3], dirNeg = stackalloc double[3];
            Scale(dirNeg, xk, 1 / xNorm);
            Scale(dir, dirNeg, -1);
            Support(v, o, out index1, out index2, obj1, obj2, dir, dirNeg);
        }

        /// <summary>
        ///     GJK (<c>gjk</c>, MuJoCo 4): from the witness guess in the status, support points toward the origin and the
        ///     signed-volume subalgorithm shrink the simplex to the part closest to the origin, the next point taken by exact
        ///     projection on the simplex's plane or line, until the Frank-Wolfe gap closes, the norm stops changing, a
        ///     tetrahedron encloses the origin, or the geoms are proved apart beyond the cutoff. The simplex's points on each
        ///     geom are left in the objects (<see cref="CcdObject.SuppVert"/>).
        /// </summary>
        /// <param name="status">The status: witness guess in; distance, witnesses and separation out.</param>
        /// <param name="obj1">Geom 1, which receives its support vertices.</param>
        /// <param name="obj2">Geom 2, likewise.</param>
        private static void Gjk(CcdStatus status, CcdObject obj1, CcdObject obj2) {
            bool getDist = status.DistCutoff > 0;
            Span<double> simplex = stackalloc double[12], vtx = stackalloc double[9];
            double[] simplex1 = obj1.SuppVert, simplex2 = obj2.SuppVert;
            int[] index1 = obj1.SuppInd, index2 = obj2.SuppInd;
            int n = 0, k = 0, kmax = status.MaxIterations;
            double[] x1k = status.X1, x2k = status.X2;
            Span<double> xk = stackalloc double[3], lambda = stackalloc double[4], xNext = stackalloc double[3];
            double tol2 = status.Tolerance * status.Tolerance;
            status.Separated = false;
            obj1.NSupp = 0;
            obj2.NSupp = 0;
            bool discrete = DiscreteGeom(obj1) && DiscreteGeom(obj2);
            double epsilon = discrete ? 0 : 0.5 * tol2;
            double minNorm = discrete || getDist ? Kernels.MinVal : status.Tolerance;
            for (int c = 0; c < 3; c++)
                xk[c] = x1k[c] - x2k[c];
            double xNorm = Norm3(xk);
            if (xNorm < Kernels.MinVal) {
                xk[0] = 1;
                xk[1] = 0;
                xk[2] = 0;
                xNorm = 1;
            }
            for (; k < kmax; k++) {
                GjkSupport(vtx, 0, out int i1, out int i2, obj1, obj2, xk, xNorm);
                int sk = 3 * n;
                vtx.Slice(0, 3).CopyTo(simplex.Slice(sk, 3));
                for (int c = 0; c < 3; c++) {
                    simplex1[sk + c] = vtx[3 + c];
                    simplex2[sk + c] = vtx[6 + c];
                }
                index1[n] = i1;
                index2[n] = i2;
                // the lower bound on the distance is lower / xNorm: a positive one proves separation
                double lower = Dot3(xk, simplex.Slice(sk, 3));
                if (!getDist) {
                    if (lower > 0) {
                        status.Separated = true;
                        status.GjkIterations = k;
                        status.Nx = 0;
                        status.Dist[0] = MaxLimit;
                        return;
                    }
                } else if (lower > 0) {
                    status.Separated = true;
                    if (status.DistCutoff < MaxLimit && lower >= status.DistCutoff * xNorm) {
                        status.GjkIterations = k;
                        status.Nx = 0;
                        status.Dist[0] = MaxLimit;
                        return;
                    }
                }
                // the Frank-Wolfe duality gap
                if (k > 0) {
                    double g = xk[0] * (xk[0] - simplex[sk]) + xk[1] * (xk[1] - simplex[sk + 1]) + xk[2] * (xk[2] - simplex[sk + 2]);
                    if (g < epsilon)
                        break;
                }
                Subdistance(lambda, n + 1, simplex);
                // keep the vertices with nonzero weight, in order
                n = 0;
                for (int i = 0; i < 4; i++) {
                    if (lambda[i] == 0)
                        continue;
                    if (n != i) {
                        simplex.Slice(3 * i, 3).CopyTo(simplex.Slice(3 * n, 3));
                        Array.Copy(simplex1, 3 * i, simplex1, 3 * n, 3);
                        Array.Copy(simplex2, 3 * i, simplex2, 3 * n, 3);
                        index1[n] = index1[i];
                        index2[n] = index2[i];
                    }
                    lambda[n++] = lambda[i];
                }
                if (n < 1) {
                    status.GjkIterations = k;
                    status.Nx = 0;
                    status.Dist[0] = MaxLimit;
                    status.Separated = true;
                    return;
                }
                // a tetrahedron holds the origin
                if (n == 4) {
                    if (!status.Separated)
                        xNorm = 0;
                    break;
                }
                // the next point: the exact projection on the simplex's plane or line, else the barycentric combination
                int fallback = 1;
                if (n == 3)
                    fallback = ProjectOriginPlane(xNext, simplex.Slice(0, 3), simplex.Slice(3, 3), simplex.Slice(6, 3));
                else if (n == 2)
                    fallback = ProjectOriginLine(xNext, simplex.Slice(0, 3), simplex.Slice(3, 3));
                if (fallback != 0)
                    Lincomb(xNext, lambda, n, simplex, 0, 3, 6, 9);
                // converged, or within tolerance for contact (a 0-simplex can't seed EPA: only stop at the origin)
                double xNextNorm = Norm3(xNext);
                double tol = n > 1 ? minNorm : Kernels.MinVal;
                if (xNextNorm < tol || (k > 0 && Math.Abs(xNextNorm - xNorm) < Kernels.MinVal)) {
                    xNext.CopyTo(xk);
                    xNorm = xNextNorm;
                    break;
                }
                xNext.CopyTo(xk);
                xNorm = xNextNorm;
            }
            if (n > 0) {
                Lincomb(x1k, lambda, n, simplex1, 0, 3, 6, 9);
                Lincomb(x2k, lambda, n, simplex2, 0, 3, 6, 9);
            }
            // an early exit: a final separation check
            if (!status.Separated && n < 4 && xNorm >= Kernels.MinVal) {
                GjkSupport(vtx, 0, out _, out _, obj1, obj2, xk, xNorm);
                if (Dot3(xk, vtx.Slice(0, 3)) > 0)
                    status.Separated = true;
            }
            status.Nx = 1;
            status.GjkIterations = k;
            obj1.NSupp = n;
            obj2.NSupp = n;
            status.Dist[0] = xNorm;
        }

        /// <summary>
        ///     The barycentric coordinates of the simplex point closest to the origin (<c>subdistance</c>, Montanari et al.):
        ///     S3D for a tetrahedron, S2D for a triangle, S1D for a segment, the vertex itself for one.
        /// </summary>
        /// <param name="lambda">Receives the four coordinates; zero for the vertices dropped.</param>
        /// <param name="n">The simplex's vertex count, 1 to 4.</param>
        /// <param name="simplex">The simplex's points, 3 values each.</param>
        private static void Subdistance(Span<double> lambda, int n, ReadOnlySpan<double> simplex) {
            lambda.Slice(0, 4).Clear();
            ReadOnlySpan<double> s1 = simplex.Slice(0, 3), s2 = simplex.Slice(3, 3), s3 = simplex.Slice(6, 3), s4 = simplex.Slice(9, 3);
            switch (n) {
                case 4:
                    S3D(lambda, s1, s2, s3, s4);
                    break;
                case 3:
                    S2D(lambda, s1, s2, s3);
                    break;
                case 2:
                    S1D(lambda, s1, s2);
                    break;
                default:
                    lambda[0] = 1;
                    break;
            }
        }

        /// <summary>
        ///     The closest point of a tetrahedron (<c>S3D</c>): the origin's own coordinates when all four cofactors share the
        ///     determinant's sign, else the best of the faces whose cofactor disagrees, by squared distance (ties keep the
        ///     earlier face).
        /// </summary>
        /// <param name="lambda">Receives the four coordinates.</param>
        /// <param name="s1">The first vertex.</param>
        /// <param name="s2">The second vertex.</param>
        /// <param name="s3">The third vertex.</param>
        /// <param name="s4">The fourth vertex.</param>
        private static void S3D(Span<double> lambda, ReadOnlySpan<double> s1, ReadOnlySpan<double> s2, ReadOnlySpan<double> s3, ReadOnlySpan<double> s4) {
            double c41 = -Det3(s2, s3, s4), c42 = Det3(s1, s3, s4), c43 = -Det3(s1, s2, s4), c44 = Det3(s1, s2, s3);
            double mDet = c41 + c42 + c43 + c44;
            int comp1 = SameSign2(mDet, c41), comp2 = SameSign2(mDet, c42), comp3 = SameSign2(mDet, c43), comp4 = SameSign2(mDet, c44);
            if (comp1 != 0 && comp2 != 0 && comp3 != 0 && comp4 != 0) {
                lambda[0] = c41 / mDet;
                lambda[1] = c42 / mDet;
                lambda[2] = c43 / mDet;
                lambda[3] = c44 / mDet;
                return;
            }
            double dmin = MaxLimit;
            Span<double> l2 = stackalloc double[3], x = stackalloc double[3], pts = stackalloc double[9];
            if (comp1 == 0) {
                S2D(l2, s2, s3, s4);
                Face3(x, l2, s2, s3, s4, pts);
                double d = Dot3(x, x);
                lambda[0] = 0;
                lambda[1] = l2[0];
                lambda[2] = l2[1];
                lambda[3] = l2[2];
                dmin = d;
            }
            if (comp2 == 0) {
                S2D(l2, s1, s3, s4);
                Face3(x, l2, s1, s3, s4, pts);
                double d = Dot3(x, x);
                if (d < dmin) {
                    lambda[0] = l2[0];
                    lambda[1] = 0;
                    lambda[2] = l2[1];
                    lambda[3] = l2[2];
                    dmin = d;
                }
            }
            if (comp3 == 0) {
                S2D(l2, s1, s2, s4);
                Face3(x, l2, s1, s2, s4, pts);
                double d = Dot3(x, x);
                if (d < dmin) {
                    lambda[0] = l2[0];
                    lambda[1] = l2[1];
                    lambda[2] = 0;
                    lambda[3] = l2[2];
                    dmin = d;
                }
            }
            if (comp4 == 0) {
                S2D(l2, s1, s2, s3);
                Face3(x, l2, s1, s2, s3, pts);
                double d = Dot3(x, x);
                if (d < dmin) {
                    lambda[0] = l2[0];
                    lambda[1] = l2[1];
                    lambda[2] = l2[2];
                    lambda[3] = 0;
                }
            }
        }

        /// <summary><c>lincomb</c> of three separate points: <c>l[0]*a + l[1]*b + l[2]*c</c>, left to right.</summary>
        /// <param name="res">Receives the point.</param>
        /// <param name="l">The weights.</param>
        /// <param name="a">The first point.</param>
        /// <param name="b">The second point.</param>
        /// <param name="c">The third point.</param>
        /// <param name="scratch">Nine values of scratch, where the three are gathered.</param>
        private static void Face3(Span<double> res, ReadOnlySpan<double> l, ReadOnlySpan<double> a, ReadOnlySpan<double> b,
                                  ReadOnlySpan<double> c, Span<double> scratch) {
            a.Slice(0, 3).CopyTo(scratch);
            b.Slice(0, 3).CopyTo(scratch.Slice(3));
            c.Slice(0, 3).CopyTo(scratch.Slice(6));
            Lincomb(res, l, 3, scratch, 0, 3, 6, 0);
        }

        /// <summary>
        ///     The closest point of a triangle (<c>S2D</c>, MuJoCo 4): the origin projected on its plane, its 2-D coordinates
        ///     on the projection that keeps the most area (minors from the edges at <paramref name="s1"/>, cofactors relative to
        ///     the projection), else the best of the edges whose cofactor disagrees; a degenerate plane tries all three edges.
        /// </summary>
        /// <param name="lambda">Receives the three coordinates.</param>
        /// <param name="s1">The first vertex.</param>
        /// <param name="s2">The second vertex.</param>
        /// <param name="s3">The third vertex.</param>
        private static void S2D(Span<double> lambda, ReadOnlySpan<double> s1, ReadOnlySpan<double> s2, ReadOnlySpan<double> s3) {
            int comp1 = 0, comp2 = 0, comp3 = 0;
            Span<double> po = stackalloc double[3];
            if (ProjectOriginPlane(po, s1, s2, s3) == 0) {
                double d21x = s2[0] - s1[0], d21y = s2[1] - s1[1], d21z = s2[2] - s1[2];
                double d31x = s3[0] - s1[0], d31y = s3[1] - s1[1], d31z = s3[2] - s1[2];
                double m14 = d21y * d31z - d21z * d31y;
                double m24 = d21x * d31z - d21z * d31x;
                double m34 = d21x * d31y - d21y * d31x;
                double mu1 = Math.Abs(m14), mu2 = Math.Abs(m24), mu3 = Math.Abs(m34);
                int x, y;
                double mMax;
                if (mu1 >= mu2 && mu1 >= mu3) {
                    mMax = m14;
                    x = 1;
                    y = 2;
                } else if (mu2 >= mu3) {
                    mMax = m24;
                    x = 0;
                    y = 2;
                } else {
                    mMax = m34;
                    x = 0;
                    y = 1;
                }
                double d1x = s1[x] - po[x], d1y = s1[y] - po[y];
                double d2x = s2[x] - po[x], d2y = s2[y] - po[y];
                double d3x = s3[x] - po[x], d3y = s3[y] - po[y];
                double c31 = d2x * d3y - d3x * d2y;
                double c32 = d3x * d1y - d1x * d3y;
                double c33 = d1x * d2y - d2x * d1y;
                comp1 = SameSign2(mMax, c31);
                comp2 = SameSign2(mMax, c32);
                comp3 = SameSign2(mMax, c33);
                if (comp1 != 0 && comp2 != 0 && comp3 != 0) {
                    double mDet = c31 + c32 + c33;
                    lambda[0] = c31 / mDet;
                    lambda[1] = c32 / mDet;
                    lambda[2] = c33 / mDet;
                    return;
                }
            }
            // each edge whose cofactor disagrees, in MuJoCo's order: the first unconditionally, the others when closer
            double dmin = MaxLimit;
            Span<double> l1 = stackalloc double[2], p = stackalloc double[3];
            if (comp1 == 0) {
                S1D(l1, s2, s3);
                for (int k = 0; k < 3; k++)
                    p[k] = l1[0] * s2[k] + l1[1] * s3[k];
                double d = Dot3(p, p);
                lambda[0] = 0;
                lambda[1] = l1[0];
                lambda[2] = l1[1];
                dmin = d;
            }
            if (comp2 == 0) {
                S1D(l1, s1, s3);
                for (int k = 0; k < 3; k++)
                    p[k] = l1[0] * s1[k] + l1[1] * s3[k];
                double d = Dot3(p, p);
                if (d < dmin) {
                    lambda[0] = l1[0];
                    lambda[1] = 0;
                    lambda[2] = l1[1];
                    dmin = d;
                }
            }
            if (comp3 == 0) {
                S1D(l1, s1, s2);
                for (int k = 0; k < 3; k++)
                    p[k] = l1[0] * s1[k] + l1[1] * s2[k];
                double d = Dot3(p, p);
                if (d < dmin) {
                    lambda[0] = l1[0];
                    lambda[1] = l1[1];
                    lambda[2] = 0;
                }
            }
        }

        /// <summary>
        ///     The closest point of a segment (<c>S1D</c>, MuJoCo 4): the origin projected on its line, its coordinate on the axis
        ///     of the longest shadow (ties go to the later axis), or the nearer end; a degenerate segment gives the second point.
        /// </summary>
        /// <param name="lambda">Receives the two coordinates.</param>
        /// <param name="s1">The first point.</param>
        /// <param name="s2">The second point.</param>
        private static void S1D(Span<double> lambda, ReadOnlySpan<double> s1, ReadOnlySpan<double> s2) {
            Span<double> po = stackalloc double[3];
            if (ProjectOriginLine(po, s1, s2) != 0) {
                lambda[0] = 0;
                lambda[1] = 1;
                return;
            }
            double mu = s1[0] - s2[0], muMax = mu;
            int index = 0;
            mu = s1[1] - s2[1];
            if (Math.Abs(mu) >= Math.Abs(muMax)) {
                muMax = mu;
                index = 1;
            }
            mu = s1[2] - s2[2];
            if (Math.Abs(mu) >= Math.Abs(muMax)) {
                muMax = mu;
                index = 2;
            }
            double c1 = po[index] - s2[index], c2 = s1[index] - po[index];
            int comp1 = SameSign2(muMax, c1), comp2 = SameSign2(muMax, c2);
            if (comp1 != 0 && comp2 != 0) {
                lambda[0] = c1 / muMax;
                lambda[1] = c2 / muMax;
            } else if (comp1 != 0) {
                lambda[0] = 1;
                lambda[1] = 0;
            } else {
                lambda[0] = 0;
                lambda[1] = 1;
            }
        }

        // ------------------------------------------------------------------------------------------ EPA's polytopes

        /// <summary>
        ///     EPA's support vertex along a face's normal (<c>epaSupport</c>), appended to the polytope; the x axis when the
        ///     normal is degenerate.
        /// </summary>
        /// <param name="pt">The polytope, which gains the vertex.</param>
        /// <param name="obj1">Geom 1, supported along the normal.</param>
        /// <param name="obj2">Geom 2, supported against it.</param>
        /// <param name="d">The normal, not normalized.</param>
        /// <param name="dnorm">The normal's norm; at or below <c>mjMINVAL</c> the x axis is used instead.</param>
        /// <returns>The new vertex's index.</returns>
        private static int EpaSupport(CcdPolytope pt, CcdObject obj1, CcdObject obj2, ReadOnlySpan<double> d, double dnorm) {
            Span<double> dir = stackalloc double[3] { 1, 0, 0 }, dirNeg = stackalloc double[3] { -1, 0, 0 };
            if (dnorm > Kernels.MinVal) {
                dir[0] = d[0] / dnorm;
                dir[1] = d[1] / dnorm;
                dir[2] = d[2] / dnorm;
                Scale(dirNeg, dir, -1);
            }
            int n = pt.Nverts++;
            Support(pt.Verts, n * VertexSize, out pt.Index1[n], out pt.Index2[n], obj1, obj2, dir, dirNeg);
            return n;
        }

        /// <summary>Copies support vertex <paramref name="i"/> of both geoms into the polytope (<c>insertVertex</c>).</summary>
        /// <param name="pt">The polytope.</param>
        /// <param name="obj1">Geom 1, holding the vertex's point on it.</param>
        /// <param name="obj2">Geom 2, likewise.</param>
        /// <param name="i">The support vertex.</param>
        /// <returns>Its index in the polytope.</returns>
        private static int InsertVertex(CcdPolytope pt, CcdObject obj1, CcdObject obj2, int i) {
            int n = pt.Nverts++, o = n * VertexSize;
            double[] v = pt.Verts;
            for (int k = 0; k < 3; k++) {
                v[o + 3 + k] = obj1.SuppVert[3 * i + k];
                v[o + 6 + k] = obj2.SuppVert[3 * i + k];
            }
            for (int k = 0; k < 3; k++)
                v[o + k] = v[o + 3 + k] - v[o + 6 + k];
            pt.Index1[n] = obj1.SuppInd[i];
            pt.Index2[n] = obj2.SuppInd[i];
            return n;
        }

        /// <summary>A polytope vertex's difference point.</summary>
        /// <param name="pt">The polytope.</param>
        /// <param name="i">The vertex.</param>
        /// <returns>Its three values.</returns>
        private static ReadOnlySpan<double> V(CcdPolytope pt, int i) => pt.Verts.AsSpan(i * VertexSize, 3);

        /// <summary>
        ///     Replaces the simplex with a polytope edge and empties the polytope (<c>replaceSimplex2</c>), before
        ///     <see cref="Polytope2"/> starts over from it.
        /// </summary>
        /// <param name="pt">The polytope, emptied.</param>
        /// <param name="obj1">Geom 1, whose support vertices become the edge's.</param>
        /// <param name="obj2">Geom 2, likewise.</param>
        /// <param name="v1">The edge's first vertex.</param>
        /// <param name="v2">Its second vertex.</param>
        private static void ReplaceSimplex2(CcdPolytope pt, CcdObject obj1, CcdObject obj2, int v1, int v2) {
            CopySupport(pt, obj1, obj2, 0, v1);
            CopySupport(pt, obj1, obj2, 1, v2);
            obj1.NSupp = 2;
            obj2.NSupp = 2;
            pt.Nfaces = pt.Nverts = pt.Nmap = 0;
        }

        /// <summary>
        ///     Replaces the simplex with a polytope face and empties the polytope (<c>replaceSimplex3</c>), before
        ///     <see cref="Polytope3"/> starts over from it.
        /// </summary>
        /// <param name="pt">The polytope, emptied.</param>
        /// <param name="obj1">Geom 1, whose support vertices become the face's.</param>
        /// <param name="obj2">Geom 2, likewise.</param>
        /// <param name="v1">The face's first vertex.</param>
        /// <param name="v2">Its second vertex.</param>
        /// <param name="v3">Its third vertex.</param>
        private static void ReplaceSimplex3(CcdPolytope pt, CcdObject obj1, CcdObject obj2, int v1, int v2, int v3) {
            CopySupport(pt, obj1, obj2, 0, v1);
            CopySupport(pt, obj1, obj2, 1, v2);
            CopySupport(pt, obj1, obj2, 2, v3);
            obj1.NSupp = 3;
            obj2.NSupp = 3;
            pt.Nfaces = pt.Nverts = pt.Nmap = 0;
        }

        /// <summary>Copies a polytope vertex's points and indices into both geoms' support vertex <paramref name="slot"/>.</summary>
        /// <param name="pt">The polytope.</param>
        /// <param name="obj1">Geom 1.</param>
        /// <param name="obj2">Geom 2.</param>
        /// <param name="slot">The support vertex written.</param>
        /// <param name="v">The polytope vertex read.</param>
        private static void CopySupport(CcdPolytope pt, CcdObject obj1, CcdObject obj2, int slot, int v) {
            Array.Copy(pt.Verts, v * VertexSize + 3, obj1.SuppVert, 3 * slot, 3);
            Array.Copy(pt.Verts, v * VertexSize + 6, obj2.SuppVert, 3 * slot, 3);
            obj1.SuppInd[slot] = pt.Index1[v];
            obj2.SuppInd[slot] = pt.Index2[v];
        }

        /// <summary>Whether the origin lies strictly between two points within <c>mjMINEPATOL</c> (<c>originOnEdge</c>).</summary>
        /// <param name="v1">The first point.</param>
        /// <param name="v2">The second point.</param>
        /// <returns>Whether the origin is on the segment's interior.</returns>
        private static bool OriginOnEdge(ReadOnlySpan<double> v1, ReadOnlySpan<double> v2) {
            Span<double> p = stackalloc double[3];
            if (Dot3(v1, v2) < 0 && ProjectOriginLine(p, v1, v2) == 0)
                return Dot3(p, p) < MinEpaTol2;
            return false;
        }

        /// <summary>
        ///     Replaces the simplex with the better of the triangles <c>(v1, va, vb)</c> and <c>(v2, vb, va)</c>
        ///     (<c>replaceBestSimplex3</c>): a degenerate one loses, else the one whose closest point to the origin is nearer
        ///     (the first on a tie).
        /// </summary>
        /// <param name="pt">The polytope, emptied.</param>
        /// <param name="obj1">Geom 1.</param>
        /// <param name="obj2">Geom 2.</param>
        /// <param name="v1i">The segment's first vertex.</param>
        /// <param name="v2i">Its second vertex.</param>
        /// <param name="vai">The first equator vertex.</param>
        /// <param name="vbi">The second equator vertex.</param>
        private static void ReplaceBestSimplex3(CcdPolytope pt, CcdObject obj1, CcdObject obj2, int v1i, int v2i, int vai, int vbi) {
            ReadOnlySpan<double> v1 = V(pt, v1i), v2 = V(pt, v2i), va = V(pt, vai), vb = V(pt, vbi);
            Span<double> e1a = stackalloc double[3], e1b = stackalloc double[3], n1 = stackalloc double[3];
            Span<double> e2a = stackalloc double[3], e2b = stackalloc double[3], n2 = stackalloc double[3];
            for (int k = 0; k < 3; k++) {
                e1a[k] = va[k] - v1[k];
                e1b[k] = vb[k] - v1[k];
            }
            Spatial.Cross(n1, e1a, e1b);
            double nn1 = Dot3(n1, n1);
            for (int k = 0; k < 3; k++) {
                e2a[k] = vb[k] - v2[k];
                e2b[k] = va[k] - v2[k];
            }
            Spatial.Cross(n2, e2a, e2b);
            double nn2 = Dot3(n2, n2);
            if (nn1 < Kernels.MinVal && nn2 >= Kernels.MinVal) {
                ReplaceSimplex3(pt, obj1, obj2, v2i, vbi, vai);
                return;
            }
            if (nn2 < Kernels.MinVal && nn1 >= Kernels.MinVal) {
                ReplaceSimplex3(pt, obj1, obj2, v1i, vai, vbi);
                return;
            }
            Span<double> l1 = stackalloc double[3], l2 = stackalloc double[3], p1 = stackalloc double[3], p2 = stackalloc double[3];
            Span<double> pts = stackalloc double[9];
            S2D(l1, v1, va, vb);
            Face3(p1, l1, v1, va, vb, pts);
            double d1 = Dot3(p1, p1);
            S2D(l2, v2, vb, va);
            Face3(p2, l2, v2, vb, va, pts);
            double d2 = Dot3(p2, p2);
            if (d2 < d1)
                ReplaceSimplex3(pt, obj1, obj2, v2i, vbi, vai);
            else
                ReplaceSimplex3(pt, obj1, obj2, v1i, vai, vbi);
        }

        /// <summary>Whether the origin and <c>p3</c> lie on the same side of the plane through the first three (<c>sameSide</c>).</summary>
        /// <param name="p0">The plane's first point.</param>
        /// <param name="p1">Its second point.</param>
        /// <param name="p2">Its third point.</param>
        /// <param name="p3">The point compared with the origin.</param>
        /// <returns>
        ///     <see langword="true"/> when both lie strictly on one side; a point on the plane gives <see langword="false"/>.
        /// </returns>
        private static bool SameSide(ReadOnlySpan<double> p0, ReadOnlySpan<double> p1, ReadOnlySpan<double> p2, ReadOnlySpan<double> p3) {
            Span<double> d1 = stackalloc double[3], d2 = stackalloc double[3], d3 = stackalloc double[3], d4 = stackalloc double[3], n = stackalloc double[3];
            for (int k = 0; k < 3; k++) {
                d1[k] = p1[k] - p0[k];
                d2[k] = p2[k] - p0[k];
            }
            Spatial.Cross(n, d1, d2);
            for (int k = 0; k < 3; k++)
                d3[k] = p3[k] - p0[k];
            double dot1 = Dot3(n, d3);
            Scale(d4, p0, -1);
            double dot2 = Dot3(n, d4);
            return (dot1 > 0 && dot2 > 0) || (dot1 < 0 && dot2 < 0);
        }

        /// <summary>Whether the origin lies strictly inside the tetrahedron (<c>testTetra</c>).</summary>
        /// <param name="p0">The first vertex.</param>
        /// <param name="p1">The second.</param>
        /// <param name="p2">The third.</param>
        /// <param name="p3">The fourth.</param>
        /// <returns><see langword="true"/> when the origin is strictly inside; on a face counts as outside.</returns>
        private static bool TestTetra(ReadOnlySpan<double> p0, ReadOnlySpan<double> p1, ReadOnlySpan<double> p2, ReadOnlySpan<double> p3) =>
            SameSide(p0, p1, p2, p3) && SameSide(p1, p2, p3, p0) && SameSide(p2, p3, p0, p1) && SameSide(p3, p0, p1, p2);

        /// <summary>
        ///     The rotation by 120° about an axis (<c>rotmat</c>), with MuJoCo's rounded <c>sin 120° = 0.86602540378</c>.
        /// </summary>
        /// <param name="r">Receives the 3x3 row-major matrix.</param>
        /// <param name="axis">The axis (normalized here).</param>
        private static void RotMat(Span<double> r, ReadOnlySpan<double> axis) {
            double n = Norm3(axis);
            double u1 = axis[0] / n, u2 = axis[1] / n, u3 = axis[2] / n;
            const double sin = 0.86602540378, cos = -0.5;
            r[0] = cos + u1 * u1 * (1 - cos);
            r[1] = u1 * u2 * (1 - cos) - u3 * sin;
            r[2] = u1 * u3 * (1 - cos) + u2 * sin;
            r[3] = u2 * u1 * (1 - cos) + u3 * sin;
            r[4] = cos + u2 * u2 * (1 - cos);
            r[5] = u2 * u3 * (1 - cos) - u1 * sin;
            r[6] = u1 * u3 * (1 - cos) - u2 * sin;
            r[7] = u2 * u3 * (1 - cos) + u1 * sin;
            r[8] = cos + u3 * u3 * (1 - cos);
        }

        /// <summary>
        ///     Whether the ray from <c>v1</c> through <c>v2</c> crosses the triangle <c>v3v4v5</c> within a tolerance
        ///     (<c>rayTriangle</c>), by the signs of three volumes.
        /// </summary>
        /// <param name="det">Receives the three volumes, or is empty to discard them.</param>
        /// <param name="v1">The ray's origin.</param>
        /// <param name="v2">A second point on the ray.</param>
        /// <param name="v3">The triangle's first vertex.</param>
        /// <param name="v4">Its second vertex.</param>
        /// <param name="v5">Its third vertex.</param>
        /// <param name="tol">The volumes' tolerance.</param>
        /// <returns>1 or -1 when all volumes reach the tolerance with one sign, 0 otherwise.</returns>
        private static int RayTriangle(Span<double> det, ReadOnlySpan<double> v1, ReadOnlySpan<double> v2, ReadOnlySpan<double> v3,
                                       ReadOnlySpan<double> v4, ReadOnlySpan<double> v5, double tol) {
            Span<double> d12 = stackalloc double[3], d13 = stackalloc double[3], d14 = stackalloc double[3], d15 = stackalloc double[3];
            Span<double> d = stackalloc double[3];
            for (int k = 0; k < 3; k++) {
                d12[k] = v2[k] - v1[k];
                d13[k] = v3[k] - v1[k];
                d14[k] = v4[k] - v1[k];
                d15[k] = v5[k] - v1[k];
            }
            d[0] = Det3(d13, d14, d12);
            d[1] = Det3(d14, d15, d12);
            d[2] = Det3(d15, d13, d12);
            // MuJoCo writes into the caller's array or a temporary; the volumes are computed once either way
            if (!det.IsEmpty)
                d.CopyTo(det);
            if (d[0] >= tol && d[1] >= tol && d[2] >= tol)
                return 1;
            if (d[0] <= -tol && d[1] <= -tol && d[2] <= -tol)
                return -1;
            return 0;
        }

        /// <summary>
        ///     The initial polytope from GJK's segment (<c>polytope2</c>, MuJoCo 4): three support points 120° apart around the
        ///     segment; an equator that does not enclose the segment is fixed by a fourth support across its weakest edge; then
        ///     a hexahedron of six faces. A face through the origin hands the problem to <see cref="Polytope3"/> on the better
        ///     neighbouring triangle, and a non-convex hexahedron stops EPA.
        /// </summary>
        /// <param name="pt">The polytope, empty on entry, built here.</param>
        /// <param name="status">The status.</param>
        /// <param name="obj1">Geom 1, holding the segment's points on it.</param>
        /// <param name="obj2">Geom 2, likewise.</param>
        /// <returns>0 to run EPA, or the EPA status that stops it.</returns>
        private static int Polytope2(CcdPolytope pt, CcdStatus status, CcdObject obj1, CcdObject obj2) {
            int v1i = InsertVertex(pt, obj1, obj2, 0), v2i = InsertVertex(pt, obj1, obj2, 1);
            Span<double> diff = stackalloc double[3];
            for (int k = 0; k < 3; k++)
                diff[k] = V(pt, v2i)[k] - V(pt, v1i)[k];
            // the coordinate axis most perpendicular to the segment (the first on ties)
            double value = MaxLimit;
            int index = 0;
            for (int i = 0; i < 3; i++) {
                if (Math.Abs(diff[i]) < value) {
                    value = Math.Abs(diff[i]);
                    index = i;
                }
            }
            Span<double> e = stackalloc double[3], d1 = stackalloc double[3], d2 = stackalloc double[3], d3 = stackalloc double[3];
            Span<double> r = stackalloc double[9];
            e[index] = 1;
            Spatial.Cross(d1, e, diff);
            RotMat(r, diff);
            Spatial.MulMatVec3(d2, r, d1);
            Spatial.MulMatVec3(d3, r, d2);
            int v3i = EpaSupport(pt, obj1, obj2, d1, Norm3(d1));
            int v4i = EpaSupport(pt, obj1, obj2, d2, Norm3(d2));
            int v5i = EpaSupport(pt, obj1, obj2, d3, Norm3(d3));
            Span<double> det = stackalloc double[3];
            double diff2 = Dot3(diff, diff);
            double tol = Kernels.MinVal * diff2 * Math.Sqrt(diff2);
            if (RayTriangle(det, V(pt, v1i), V(pt, v2i), V(pt, v3i), V(pt, v4i), V(pt, v5i), tol) == 0) {
                // the equator edge (va, vb) with the smallest volume, which the segment lies on or outside of
                int vai = v5i, vbi = v3i, vci = v4i;
                if (det[0] <= det[1] && det[0] <= det[2]) {
                    vai = v3i;
                    vbi = v4i;
                    vci = v5i;
                } else if (det[1] <= det[0] && det[1] <= det[2]) {
                    vai = v4i;
                    vbi = v5i;
                    vci = v3i;
                }
                // the plane's outward normal away from vc, tilted by 1e-3 along the segment toward vc's pole
                Span<double> edge = stackalloc double[3], d4 = stackalloc double[3], d1c = stackalloc double[3];
                for (int k = 0; k < 3; k++) {
                    edge[k] = V(pt, vbi)[k] - V(pt, vai)[k];
                    d1c[k] = V(pt, vci)[k] - V(pt, v1i)[k];
                }
                Spatial.Cross(d4, edge, diff);
                double sign = Dot3(d1c, diff) >= 0.5 * diff2 ? 1e-3 : -1e-3;
                double s = sign * Norm3(edge);
                for (int k = 0; k < 3; k++)
                    d4[k] = d4[k] + s * diff[k];
                // the new support is popped so it can overwrite an equator vertex in place
                int v6i = EpaSupport(pt, obj1, obj2, d4, Norm3(d4));
                pt.Nverts--;
                Span<double> v6 = stackalloc double[VertexSize];
                pt.Verts.AsSpan(v6i * VertexSize, VertexSize).CopyTo(v6);
                int v6Index1 = pt.Index1[v6i], v6Index2 = pt.Index2[v6i];
                int replace = -1;
                if (RayTriangle([], V(pt, v1i), V(pt, v2i), v6.Slice(0, 3), V(pt, vbi), V(pt, vci), tol) != 0)
                    replace = vai;
                else if (RayTriangle([], V(pt, v1i), V(pt, v2i), V(pt, vai), v6.Slice(0, 3), V(pt, vci), tol) != 0)
                    replace = vbi;
                else if (RayTriangle([], V(pt, v1i), V(pt, v2i), V(pt, vai), V(pt, vbi), v6.Slice(0, 3), tol) != 0)
                    replace = vci;
                if (replace >= 0) {
                    v6.CopyTo(pt.Verts.AsSpan(replace * VertexSize, VertexSize));
                    pt.Index1[replace] = v6Index1;
                    pt.Index2[replace] = v6Index2;
                }
            }
            for (int i = 0; i < 3; i++)
                pt.Center[i] = 0.2 * (V(pt, v1i)[i] + V(pt, v2i)[i] + V(pt, v3i)[i] + V(pt, v4i)[i] + V(pt, v5i)[i]);
            // the hexahedron; a face through the origin falls back to the better triangle at it
            if (AttachFace(pt, v1i, v3i, v4i, 1, 3, 2) < MinDist) {
                ReplaceBestSimplex3(pt, obj1, obj2, v1i, v2i, v3i, v4i);
                return Polytope3(pt, status, obj1, obj2, false);
            }
            if (AttachFace(pt, v1i, v5i, v3i, 2, 4, 0) < MinDist) {
                ReplaceBestSimplex3(pt, obj1, obj2, v1i, v2i, v5i, v3i);
                return Polytope3(pt, status, obj1, obj2, false);
            }
            if (AttachFace(pt, v1i, v4i, v5i, 0, 5, 1) < MinDist) {
                ReplaceBestSimplex3(pt, obj1, obj2, v1i, v2i, v4i, v5i);
                return Polytope3(pt, status, obj1, obj2, false);
            }
            if (AttachFace(pt, v2i, v4i, v3i, 5, 0, 4) < MinDist) {
                ReplaceBestSimplex3(pt, obj1, obj2, v1i, v2i, v3i, v4i);
                return Polytope3(pt, status, obj1, obj2, false);
            }
            if (AttachFace(pt, v2i, v3i, v5i, 3, 1, 5) < MinDist) {
                ReplaceBestSimplex3(pt, obj1, obj2, v1i, v2i, v5i, v3i);
                return Polytope3(pt, status, obj1, obj2, false);
            }
            if (AttachFace(pt, v2i, v5i, v4i, 4, 2, 3) < MinDist) {
                ReplaceBestSimplex3(pt, obj1, obj2, v1i, v2i, v4i, v5i);
                return Polytope3(pt, status, obj1, obj2, false);
            }
            // a non-convex hexahedron: a face near the origin still recovers through polytope3
            if (RayTriangle([], V(pt, v1i), V(pt, v2i), V(pt, v3i), V(pt, v4i), V(pt, v5i), 0) == 0) {
                for (int i = 0; i < 6; i++) {
                    if (pt.Faces[i].Dist2 < MinEpaTol2) {
                        int vai = i % 3 == 0 ? v3i : i % 3 == 1 ? v5i : v4i;
                        int vbi = i % 3 == 0 ? v4i : i % 3 == 1 ? v3i : v5i;
                        ReplaceBestSimplex3(pt, obj1, obj2, v1i, v2i, vai, vbi);
                        return Polytope3(pt, status, obj1, obj2, false);
                    }
                }
                return P2Nonconvex;
            }
            for (int i = 0; i < 6; i++) {
                pt.Map[i] = i;
                pt.Faces[i].Index = i;
            }
            pt.Nmap = 6;
            return 0;
        }

        /// <summary>
        ///     The affine coordinates of a point in a triangle's plane (<c>triAffineCoord</c>, MuJoCo 4): signed areas, relative
        ///     to the point, on the coordinate plane that keeps the most of the triangle, over their sum.
        /// </summary>
        /// <param name="lambda">Receives the three coordinates.</param>
        /// <param name="v1">The triangle's first vertex.</param>
        /// <param name="v2">Its second vertex.</param>
        /// <param name="v3">Its third vertex.</param>
        /// <param name="p">The point, in the triangle's plane.</param>
        private static void TriAffineCoord(Span<double> lambda, ReadOnlySpan<double> v1, ReadOnlySpan<double> v2, ReadOnlySpan<double> v3,
                                           ReadOnlySpan<double> p) {
            double d21x = v2[0] - v1[0], d21y = v2[1] - v1[1], d21z = v2[2] - v1[2];
            double d31x = v3[0] - v1[0], d31y = v3[1] - v1[1], d31z = v3[2] - v1[2];
            double m14 = d21y * d31z - d21z * d31y;
            double m24 = d21x * d31z - d21z * d31x;
            double m34 = d21x * d31y - d21y * d31x;
            int x, y;
            double mu1 = Math.Abs(m14), mu2 = Math.Abs(m24), mu3 = Math.Abs(m34);
            if (mu1 >= mu2 && mu1 >= mu3) {
                x = 1;
                y = 2;
            } else if (mu2 >= mu3) {
                x = 0;
                y = 2;
            } else {
                x = 0;
                y = 1;
            }
            double d1x = v1[x] - p[x], d1y = v1[y] - p[y];
            double d2x = v2[x] - p[x], d2y = v2[y] - p[y];
            double d3x = v3[x] - p[x], d3y = v3[y] - p[y];
            double c31 = d2x * d3y - d3x * d2y;
            double c32 = d3x * d1y - d1x * d3y;
            double c33 = d1x * d2y - d2x * d1y;
            double mDet = c31 + c32 + c33;
            lambda[0] = c31 / mDet;
            lambda[1] = c32 / mDet;
            lambda[2] = c33 / mDet;
        }

        /// <summary>
        ///     Whether a point lies in a triangle (<c>triPointIntersect</c>): non-negative affine coordinates, and the point
        ///     they rebuild within <c>mjMINVAL</c> of it.
        /// </summary>
        /// <param name="v1">The triangle's first vertex.</param>
        /// <param name="v2">Its second vertex.</param>
        /// <param name="v3">Its third vertex.</param>
        /// <param name="p">The point.</param>
        /// <returns><see langword="true"/> when the point is in the triangle, edges included.</returns>
        private static bool TriPointIntersect(ReadOnlySpan<double> v1, ReadOnlySpan<double> v2, ReadOnlySpan<double> v3, ReadOnlySpan<double> p) {
            Span<double> lambda = stackalloc double[3], diff = stackalloc double[3];
            TriAffineCoord(lambda, v1, v2, v3, p);
            if (lambda[0] < 0 || lambda[1] < 0 || lambda[2] < 0)
                return false;
            for (int k = 0; k < 3; k++)
                diff[k] = v1[k] * lambda[0] + v2[k] * lambda[1] + v3[k] * lambda[2] - p[k];
            return Norm3(diff) < Kernels.MinVal;
        }

        /// <summary>
        ///     Recovers a convex hexahedron or tetrahedron when the ray between the poles misses the triangle's interior
        ///     (<c>recoverHexa</c>): one positive volume swaps poles and equator; otherwise the most negative volume picks the
        ///     outer tetrahedron.
        /// </summary>
        /// <param name="idx">The vertex indices <c>{v1, v2, v3, v4, v5}</c>, reordered in place.</param>
        /// <param name="det">The volumes of <see cref="RayTriangle"/>.</param>
        /// <returns><see langword="true"/> for a hexahedron, <see langword="false"/> for a tetrahedron.</returns>
        private static bool RecoverHexa(Span<int> idx, ReadOnlySpan<double> det) {
            int i1 = idx[0], i2 = idx[1], i3 = idx[2], i4 = idx[3], i5 = idx[4];
            if (det[0] > 0 && det[1] < -Kernels.MinVal && det[2] < -Kernels.MinVal) {
                idx[0] = i3;
                idx[1] = i5;
                idx[2] = i4;
                idx[3] = i1;
                idx[4] = i2;
                return true;
            }
            if (det[1] > 0 && det[2] < -Kernels.MinVal && det[0] < -Kernels.MinVal) {
                idx[0] = i1;
                idx[1] = i5;
                idx[2] = i4;
                idx[3] = i2;
                idx[4] = i3;
                return true;
            }
            if (det[2] > 0 && det[0] < -Kernels.MinVal && det[1] < -Kernels.MinVal) {
                idx[0] = i2;
                idx[1] = i5;
                idx[2] = i4;
                idx[3] = i3;
                idx[4] = i1;
                return true;
            }
            if (det[0] <= det[1] && det[0] <= det[2]) {
                idx[0] = i1;
                idx[1] = i2;
            } else if (det[1] <= det[0] && det[1] <= det[2]) {
                idx[0] = i2;
                idx[1] = i3;
            } else {
                idx[0] = i3;
                idx[1] = i1;
            }
            return false;
        }

        /// <summary>
        ///     The initial polytope from GJK's triangle (<c>polytope3</c>, MuJoCo 4): the support points against and along its
        ///     normal (in that order) close a hexahedron of six faces, or the recovered tetrahedron when the poles miss the
        ///     triangle; a degenerate normal, a pole inside or coplanar with the triangle, a face through the origin (handed to
        ///     <see cref="Polytope2"/> when allowed and the origin is on an edge) or a polytope missing the origin stops EPA.
        /// </summary>
        /// <param name="pt">The polytope, empty on entry, built here.</param>
        /// <param name="status">The status.</param>
        /// <param name="obj1">Geom 1, holding the triangle's points on it.</param>
        /// <param name="obj2">Geom 2, likewise.</param>
        /// <param name="allowP2">Whether a face through the origin may fall back to <see cref="Polytope2"/>.</param>
        /// <returns>0 to run EPA, or the EPA status that stops it.</returns>
        private static int Polytope3(CcdPolytope pt, CcdStatus status, CcdObject obj1, CcdObject obj2, bool allowP2) {
            int v1i = InsertVertex(pt, obj1, obj2, 0), v2i = InsertVertex(pt, obj1, obj2, 1), v3i = InsertVertex(pt, obj1, obj2, 2);
            Span<double> diff1 = stackalloc double[3], diff2 = stackalloc double[3], n = stackalloc double[3], nNeg = stackalloc double[3];
            for (int k = 0; k < 3; k++) {
                diff1[k] = V(pt, v2i)[k] - V(pt, v1i)[k];
                diff2[k] = V(pt, v3i)[k] - V(pt, v1i)[k];
            }
            Spatial.Cross(n, diff1, diff2);
            double nNorm = Norm3(n);
            if (nNorm < Kernels.MinVal)
                return P3BadNormal;
            Scale(nNeg, n, -1);
            int v5i = EpaSupport(pt, obj1, obj2, nNeg, nNorm);
            int v4i = EpaSupport(pt, obj1, obj2, n, nNorm);
            if (TriPointIntersect(V(pt, v1i), V(pt, v2i), V(pt, v3i), V(pt, v4i)))
                return P3InvalidV4;
            if (TriPointIntersect(V(pt, v1i), V(pt, v2i), V(pt, v3i), V(pt, v5i)))
                return P3InvalidV5;
            // the poles must leave the triangle's plane
            Span<double> d4 = stackalloc double[3], d5 = stackalloc double[3];
            for (int k = 0; k < 3; k++) {
                d4[k] = V(pt, v4i)[k] - V(pt, v1i)[k];
                d5[k] = V(pt, v5i)[k] - V(pt, v1i)[k];
            }
            if (Dot3(n, d4) < MinEpaTol * nNorm || Dot3(nNeg, d5) < MinEpaTol * nNorm)
                return P3OriginOnFace;
            // a non-convex hexahedron is recovered into a convex one or a tetrahedron
            Span<double> det = stackalloc double[3];
            Span<int> idx = stackalloc int[5] { v1i, v2i, v3i, v4i, v5i };
            bool isHexa = true;
            if (RayTriangle(det, V(pt, v4i), V(pt, v5i), V(pt, v1i), V(pt, v2i), V(pt, v3i), 0) == 0)
                isHexa = RecoverHexa(idx, det);
            ReadOnlySpan<double> p1 = V(pt, idx[0]), p2 = V(pt, idx[1]), p3 = V(pt, idx[2]), p4 = V(pt, idx[3]), p5 = V(pt, idx[4]);
            double dist0, dist1, dist2, dist3, dist4 = MaxLimit, dist5 = MaxLimit;
            if (isHexa) {
                for (int i = 0; i < 3; i++)
                    pt.Center[i] = 0.2 * (p1[i] + p2[i] + p3[i] + p4[i] + p5[i]);
                dist0 = AttachFace(pt, idx[3], idx[0], idx[1], 1, 3, 2);
                dist1 = AttachFace(pt, idx[3], idx[2], idx[0], 2, 4, 0);
                dist2 = AttachFace(pt, idx[3], idx[1], idx[2], 0, 5, 1);
                dist3 = AttachFace(pt, idx[4], idx[1], idx[0], 5, 0, 4);
                dist4 = AttachFace(pt, idx[4], idx[0], idx[2], 3, 1, 5);
                dist5 = AttachFace(pt, idx[4], idx[2], idx[1], 4, 2, 3);
            } else {
                // add3, add3, add3, scl3: ((v1 + v2) + v4) + v5, times 0.25
                for (int i = 0; i < 3; i++)
                    pt.Center[i] = 0.25 * (p1[i] + p2[i] + p4[i] + p5[i]);
                dist0 = AttachFace(pt, idx[0], idx[1], idx[3], 1, 3, 2);
                dist1 = AttachFace(pt, idx[0], idx[4], idx[1], 2, 3, 0);
                dist2 = AttachFace(pt, idx[0], idx[3], idx[4], 0, 3, 1);
                dist3 = AttachFace(pt, idx[4], idx[3], idx[1], 2, 0, 1);
            }
            if (dist0 < MinDist || dist1 < MinDist || dist2 < MinDist || dist3 < MinDist || dist4 < MinDist || dist5 < MinDist) {
                if (allowP2) {
                    if (OriginOnEdge(V(pt, v1i), V(pt, v2i))) {
                        ReplaceSimplex2(pt, obj1, obj2, v1i, v2i);
                        return Polytope2(pt, status, obj1, obj2);
                    }
                    if (OriginOnEdge(V(pt, v2i), V(pt, v3i))) {
                        ReplaceSimplex2(pt, obj1, obj2, v2i, v3i);
                        return Polytope2(pt, status, obj1, obj2);
                    }
                    if (OriginOnEdge(V(pt, v3i), V(pt, v1i))) {
                        ReplaceSimplex2(pt, obj1, obj2, v3i, v1i);
                        return Polytope2(pt, status, obj1, obj2);
                    }
                }
                return P3OriginOnFace;
            }
            // the polytope must hold the origin (faces within the tolerance pass)
            double tol2 = status.Tolerance * status.Tolerance;
            if (isHexa) {
                if ((!SameSide(p4, p1, p2, p3) && dist0 >= tol2) || (!SameSide(p4, p3, p1, p2) && dist1 >= tol2)
                    || (!SameSide(p4, p2, p3, p1) && dist2 >= tol2) || (!SameSide(p5, p2, p1, p3) && dist3 >= tol2)
                    || (!SameSide(p5, p1, p3, p2) && dist4 >= tol2) || (!SameSide(p5, p3, p2, p1) && dist5 >= tol2))
                    return P3MissingOrigin;
            } else {
                if ((!SameSide(p1, p2, p4, p5) && dist0 >= tol2) || (!SameSide(p1, p5, p2, p4) && dist1 >= tol2)
                    || (!SameSide(p1, p4, p5, p2) && dist2 >= tol2) || (!SameSide(p5, p4, p2, p1) && dist3 >= tol2))
                    return P3MissingOrigin;
            }
            for (int i = 0; i < pt.Nfaces; i++) {
                pt.Map[i] = i;
                pt.Faces[i].Index = i;
            }
            pt.Nmap = pt.Nfaces;
            return 0;
        }

        /// <summary>
        ///     The initial polytope from GJK's tetrahedron (<c>polytope4</c>, MuJoCo 4): its four faces; a face through the
        ///     origin, or a tetrahedron that numerically misses it with a face within <c>mjMINEPATOL</c>, starts
        ///     <see cref="Polytope3"/> over from that face.
        /// </summary>
        /// <param name="pt">The polytope, empty on entry, built here.</param>
        /// <param name="status">The status.</param>
        /// <param name="obj1">Geom 1, holding the tetrahedron's points on it.</param>
        /// <param name="obj2">Geom 2, likewise.</param>
        /// <returns>0 to run EPA, or the EPA status that stops it.</returns>
        private static int Polytope4(CcdPolytope pt, CcdStatus status, CcdObject obj1, CcdObject obj2) {
            int v1 = InsertVertex(pt, obj1, obj2, 0), v2 = InsertVertex(pt, obj1, obj2, 1);
            int v3 = InsertVertex(pt, obj1, obj2, 2), v4 = InsertVertex(pt, obj1, obj2, 3);
            // add3, add3, add3, scl3: ((v1 + v2) + v3) + v4, times 0.25
            for (int i = 0; i < 3; i++)
                pt.Center[i] = V(pt, v1)[i] + V(pt, v2)[i];
            for (int i = 0; i < 3; i++)
                pt.Center[i] = pt.Center[i] + V(pt, v3)[i];
            for (int i = 0; i < 3; i++)
                pt.Center[i] = pt.Center[i] + V(pt, v4)[i];
            for (int i = 0; i < 3; i++)
                pt.Center[i] = 0.25 * pt.Center[i];
            double dist0 = AttachFace(pt, v1, v2, v3, 1, 3, 2);
            if (dist0 < MinDist) {
                ReplaceSimplex3(pt, obj1, obj2, v1, v2, v3);
                return Polytope3(pt, status, obj1, obj2, true);
            }
            double dist1 = AttachFace(pt, v1, v4, v2, 2, 3, 0);
            if (dist1 < MinDist) {
                ReplaceSimplex3(pt, obj1, obj2, v1, v4, v2);
                return Polytope3(pt, status, obj1, obj2, true);
            }
            double dist2 = AttachFace(pt, v1, v3, v4, 0, 3, 1);
            if (dist2 < MinDist) {
                ReplaceSimplex3(pt, obj1, obj2, v1, v3, v4);
                return Polytope3(pt, status, obj1, obj2, true);
            }
            double dist3 = AttachFace(pt, v4, v3, v2, 2, 0, 1);
            if (dist3 < MinDist) {
                ReplaceSimplex3(pt, obj1, obj2, v4, v3, v2);
                return Polytope3(pt, status, obj1, obj2, true);
            }
            if (!TestTetra(V(pt, v1), V(pt, v2), V(pt, v3), V(pt, v4))) {
                double dmin = dist0;
                int t1 = v1, t2 = v2, t3 = v3;
                if (dist1 < dmin) {
                    dmin = dist1;
                    t1 = v1;
                    t2 = v4;
                    t3 = v2;
                }
                if (dist2 < dmin) {
                    dmin = dist2;
                    t1 = v1;
                    t2 = v3;
                    t3 = v4;
                }
                if (dist3 < dmin) {
                    dmin = dist3;
                    t1 = v4;
                    t2 = v3;
                    t3 = v2;
                }
                if (dmin < MinEpaTol2) {
                    ReplaceSimplex3(pt, obj1, obj2, t1, t2, t3);
                    return Polytope3(pt, status, obj1, obj2, true);
                }
                return P4MissingOrigin;
            }
            for (int i = 0; i < 4; i++) {
                pt.Map[i] = i;
                pt.Faces[i].Index = i;
            }
            pt.Nmap = 4;
            return 0;
        }

        // ------------------------------------------------------------------------------------------ EPA

        /// <summary>
        ///     Appends a face (<c>attachFace</c>): its vertices packed 10 bits each, its neighbours, the origin's projection on
        ///     its plane turned outward from the polytope's centre, and the projection's squared norm.
        /// </summary>
        /// <param name="pt">The polytope, which gains the face.</param>
        /// <param name="v1">The face's first vertex index.</param>
        /// <param name="v2">Its second vertex index.</param>
        /// <param name="v3">Its third vertex index.</param>
        /// <param name="adj1">The neighbouring face across edge 0, from the first vertex to the second.</param>
        /// <param name="adj2">Across edge 1, from the second vertex to the third.</param>
        /// <param name="adj3">Across edge 2, from the third vertex back to the first.</param>
        /// <returns>The squared distance of the face's plane to the origin; 0 for a degenerate face.</returns>
        private static double AttachFace(CcdPolytope pt, int v1, int v2, int v3, int adj1, int adj2, int adj3) {
            ref CcdFace face = ref pt.Faces[pt.Nfaces++];
            face.Verts = v1 + (v2 << 10) + (v3 << 20);
            face.Adj0 = adj1;
            face.Adj1 = adj2;
            face.Adj2 = adj3;
            Span<double> projection = stackalloc double[3];
            // MuJoCo projects with the vertices reversed (v3, v2, v1)
            if (ProjectOriginPlane(projection, V(pt, v3), V(pt, v2), V(pt, v1)) != 0)
                return 0;
            Span<double> outward = stackalloc double[3];
            for (int k = 0; k < 3; k++)
                outward[k] = V(pt, v1)[k] - pt.Center[k];
            if (Dot3(projection, outward) < 0)
                Scale(projection, projection, -1);
            face.V0 = projection[0];
            face.V1 = projection[1];
            face.V2 = projection[2];
            face.Dist2 = Dot3(projection, projection);
            face.Index = -1;
            return face.Dist2;
        }

        /// <summary>A face's vertex at slot 0, 1 or 2 of its packed indices (<c>EPA_VERT_EXPAND</c>).</summary>
        /// <param name="verts">The face's packed vertex indices.</param>
        /// <param name="slot">The slot, 0 to 2.</param>
        /// <returns>The vertex index.</returns>
        private static int Vert(int verts, int slot) => slot switch {
            0 => verts & 0x3FF,
            1 => verts >> 10 & 0x3FF,
            _ => verts >> 20 & 0x3FF,
        };

        /// <summary>A face's neighbour across edge 0, 1 or 2.</summary>
        /// <param name="f">The face.</param>
        /// <param name="edge">The edge, 0 to 2.</param>
        /// <returns>The neighbouring face's index.</returns>
        private static int Adj(in CcdFace f, int edge) => edge == 0 ? f.Adj0 : edge == 1 ? f.Adj1 : f.Adj2;

        /// <summary>Sets a face's neighbour across an edge.</summary>
        /// <param name="f">The face, changed in place.</param>
        /// <param name="edge">The edge, 0 to 2.</param>
        /// <param name="value">The neighbouring face's index.</param>
        private static void SetAdj(ref CcdFace f, int edge, int value) {
            if (edge == 0)
                f.Adj0 = value;
            else if (edge == 1)
                f.Adj1 = value;
            else
                f.Adj2 = value;
        }

        /// <summary>Removes a face from the candidate map and marks it deleted (<c>deleteFace</c>).</summary>
        /// <param name="pt">The polytope.</param>
        /// <param name="face">The face's index.</param>
        private static void DeleteFace(CcdPolytope pt, int face) {
            int index = pt.Faces[face].Index;
            if (index >= 0) {
                pt.Map[index] = pt.Map[--pt.Nmap];
                pt.Faces[pt.Map[index]].Index = index;
            }
            pt.Faces[face].Index = -2;
        }

        /// <summary>The edge of a face that starts at a vertex (<c>getEdge</c>).</summary>
        /// <param name="face">The face.</param>
        /// <param name="vertex">The vertex index the edge starts at; a vertex not on the face gives edge 2.</param>
        /// <returns>The edge, 0 to 2.</returns>
        private static int GetEdge(in CcdFace face, int vertex) =>
            Vert(face.Verts, 0) == vertex ? 0 : Vert(face.Verts, 1) == vertex ? 1 : 2;

        /// <summary>Appends an edge to the horizon (<c>addEdge</c>): the count always grows, the edge is kept while it fits.</summary>
        /// <param name="pt">The polytope holding the horizon.</param>
        /// <param name="index">The face that stays: invisible from the new vertex, it borders the deleted region.</param>
        /// <param name="edge">Its edge on the horizon.</param>
        /// <returns>Whether the edge fit.</returns>
        private static bool AddEdge(CcdPolytope pt, int index, int edge) {
            int i = pt.Nedges++;
            if (i < pt.MaxEdges) {
                pt.HorizonEdges[i] = edge;
                pt.HorizonIndices[i] = index;
                return true;
            }
            return false;
        }

        /// <summary>
        ///     The horizon's recursive step (<c>horizonRec</c>, MuJoCo 4): a face visible from the new vertex (by more than
        ///     <c>mjMINEPATOL</c> times its distance) is deleted and its other two neighbours searched; an invisible one ends the
        ///     search on the horizon.
        /// </summary>
        /// <param name="pt">The polytope; its new vertex is at <see cref="CcdPolytope.HorizonW"/>.</param>
        /// <param name="face">The face to test.</param>
        /// <param name="e">The edge of <paramref name="face"/> the search came across.</param>
        /// <returns>1 when the face was visible, 0 when not, -1 when the horizon overflowed.</returns>
        private static int HorizonRec(CcdPolytope pt, int face, int e) {
            ref CcdFace f = ref pt.Faces[face];
            double tol = MinEpaTol * Math.Sqrt(f.Dist2);
            double visTol = tol > Kernels.MinVal ? tol : Kernels.MinVal;
            int w = pt.HorizonW;
            if (f.V0 * pt.Verts[w] + f.V1 * pt.Verts[w + 1] + f.V2 * pt.Verts[w + 2] - f.Dist2 > visTol) {
                int verts = f.Verts;
                DeleteFace(pt, face);
                int ret = 1;
                for (int k = 1; k < 3; k++) {
                    int i = (e + k) % 3;
                    int adj = Adj(pt.Faces[face], i);
                    if (pt.Faces[adj].Index > -2) {
                        int adjEdge = GetEdge(pt.Faces[adj], Vert(verts, (i + 1) % 3));
                        int r = HorizonRec(pt, adj, adjEdge);
                        if (r < 0 || (r == 0 && !AddEdge(pt, adj, adjEdge)))
                            ret = -1;
                    }
                }
                return ret;
            }
            return 0;
        }

        /// <summary>
        ///     The horizon seen from the new vertex, starting at the face it was found from (<c>horizon</c>): that face is
        ///     deleted, and its three neighbours are searched in edge order.
        /// </summary>
        /// <param name="pt">The polytope; its new vertex is at <see cref="CcdPolytope.HorizonW"/>, its horizon is rebuilt.</param>
        /// <param name="face">The face the new vertex was found from.</param>
        /// <returns>0, or -1 when the horizon overflowed.</returns>
        private static int Horizon(CcdPolytope pt, int face) {
            DeleteFace(pt, face);
            int verts = pt.Faces[face].Verts;
            int ret = 0;
            for (int i = 0; i < 3; i++) {
                int adj = Adj(pt.Faces[face], i);
                if (pt.Faces[adj].Index > -2) {
                    int adjEdge = GetEdge(pt.Faces[adj], Vert(verts, (i + 1) % 3));
                    int r = HorizonRec(pt, adj, adjEdge);
                    if (r < 0 || (r == 0 && !AddEdge(pt, adj, adjEdge)))
                        ret = -1;
                }
            }
            return ret;
        }

        /// <summary>
        ///     The witness points of the final face (<c>epaWitness</c>): the origin's projection in the face's affine
        ///     coordinates, applied to the geoms' points.
        /// </summary>
        /// <param name="pt">The polytope.</param>
        /// <param name="face">The final face.</param>
        /// <param name="x1">Receives geom 1's witness point.</param>
        /// <param name="x2">Receives geom 2's witness point.</param>
        /// <returns>Minus the face's distance to the origin (the penetration).</returns>
        private static double EpaWitness(CcdPolytope pt, int face, double[] x1, double[] x2) {
            ref CcdFace f = ref pt.Faces[face];
            int a = Vert(f.Verts, 0) * VertexSize, b = Vert(f.Verts, 1) * VertexSize, c = Vert(f.Verts, 2) * VertexSize;
            Span<double> lambda = stackalloc double[3], p = stackalloc double[3] { f.V0, f.V1, f.V2 };
            TriAffineCoord(lambda, pt.Verts.AsSpan(a, 3), pt.Verts.AsSpan(b, 3), pt.Verts.AsSpan(c, 3), p);
            Lincomb(x1, lambda, 3, pt.Verts, a + 3, b + 3, c + 3, 0);
            Lincomb(x2, lambda, 3, pt.Verts, a + 6, b + 6, c + 6, 0);
            return -Math.Sqrt(f.Dist2);
        }

        /// <summary>
        ///     The expanding polytope (<c>epa</c>, MuJoCo 4): the face nearest the origin is pushed out by a support point
        ///     along its normal, the faces it can see are replaced by a fan around the new vertex, until the upper and lower
        ///     depth bounds meet within the tolerance; the final face's points become both geoms' support vertices.
        /// </summary>
        /// <param name="status">
        ///     The status: the tolerance and iteration bound in, the distance, witnesses and iterations out.
        /// </param>
        /// <param name="pt">The initial polytope, expanded in place.</param>
        /// <param name="obj1">Geom 1.</param>
        /// <param name="obj2">Geom 2.</param>
        /// <returns>The face whose plane gives the penetration (witnesses written to the status), or -1.</returns>
        private static int Epa(CcdStatus status, CcdPolytope pt, CcdObject obj1, CcdObject obj2) {
            double upper = MaxLimit, upper2 = MaxLimit, lower2;
            int face = -1, pface;
            bool discrete = DiscreteGeom(obj1) && DiscreteGeom(obj2);
            double tolerance = discrete ? MinEpaTol : status.Tolerance;
            int k, kmax = status.MaxIterations < 1000 ? status.MaxIterations : 1000;
            Span<double> fv = stackalloc double[3];
            for (k = 0; k < kmax; k++) {
                pface = face;
                lower2 = MaxLimit;
                for (int i = 0; i < pt.Nmap; i++) {
                    if (pt.Faces[pt.Map[i]].Dist2 < lower2) {
                        face = pt.Map[i];
                        lower2 = pt.Faces[face].Dist2;
                    }
                }
                if (lower2 > upper2 || face < 0) {
                    face = pface;
                    break;
                }
                // the origin on a face's plane: MuJoCo warns and stops
                if (lower2 <= 0)
                    break;
                double lower = Math.Sqrt(lower2);
                fv[0] = pt.Faces[face].V0;
                fv[1] = pt.Faces[face].V1;
                fv[2] = pt.Faces[face].V2;
                int wi = EpaSupport(pt, obj1, obj2, fv, lower);
                int w = wi * VertexSize;
                double upperK = (fv[0] * pt.Verts[w] + fv[1] * pt.Verts[w + 1] + fv[2] * pt.Verts[w + 2]) / lower;
                if (upperK < upper) {
                    upper = upperK;
                    upper2 = upper * upper;
                }
                if (upperK - lower < tolerance) {
                    // GJK can stop within tolerance with the origin just outside the polytope: no contact then
                    if (k == 0 && upper < lower - tolerance) {
                        face = -1;
                        status.EpaStatus = EpaNoContact;
                    }
                    break;
                }
                // a repeated support point ends a discrete search
                if (discrete) {
                    int i = 0, nverts = pt.Nverts - 1;
                    for (; i < nverts; i++)
                        if (pt.Index1[wi] == pt.Index1[i] && pt.Index2[wi] == pt.Index2[i])
                            break;
                    if (i != nverts)
                        break;
                }
                pt.HorizonW = w;
                // a horizon that overflows its edges: MuJoCo warns and gives up the contact
                if (Horizon(pt, face) < 0) {
                    face = -1;
                    break;
                }
                if (pt.Nedges < 3) {
                    face = -1;
                    break;
                }
                int nfaces = pt.Nfaces, nedges = pt.Nedges;
                // out of faces: MuJoCo warns and keeps the current face
                if (nedges > pt.MaxFaces - pt.Nfaces)
                    break;
                int hznIndex = pt.HorizonIndices[0], hznEdge = pt.HorizonEdges[0];
                int v1 = Vert(pt.Faces[hznIndex].Verts, hznEdge), v2 = Vert(pt.Faces[hznIndex].Verts, (hznEdge + 1) % 3);
                SetAdj(ref pt.Faces[hznIndex], hznEdge, nfaces);
                double dist2 = AttachFace(pt, wi, v2, v1, nfaces + nedges - 1, hznIndex, nfaces + 1);
                if (dist2 == 0)
                    break;
                if (dist2 >= lower2 && dist2 <= upper2) {
                    int i = pt.Nmap++;
                    pt.Map[i] = pt.Nfaces - 1;
                    pt.Faces[pt.Nfaces - 1].Index = i;
                }
                for (int i = 1; i < nedges; i++) {
                    int cur = nfaces + i, next = nfaces + (i + 1) % nedges;
                    hznIndex = pt.HorizonIndices[i];
                    hznEdge = pt.HorizonEdges[i];
                    v1 = Vert(pt.Faces[hznIndex].Verts, hznEdge);
                    v2 = Vert(pt.Faces[hznIndex].Verts, (hznEdge + 1) % 3);
                    SetAdj(ref pt.Faces[hznIndex], hznEdge, cur);
                    dist2 = AttachFace(pt, wi, v2, v1, cur - 1, hznIndex, next);
                    if (dist2 == 0) {
                        pt.Nmap = 0;
                        break;
                    }
                    if (dist2 >= lower2 && dist2 <= upper2) {
                        int idx = pt.Nmap++;
                        pt.Map[idx] = pt.Nfaces - 1;
                        pt.Faces[pt.Nfaces - 1].Index = idx;
                    }
                }
                pt.Nedges = 0;
                if (pt.Nmap == 0 || face < 0)
                    break;
            }
            status.EpaIterations = k;
            if (face >= 0) {
                status.Dist[0] = EpaWitness(pt, face, status.X1, status.X2);
                status.Nx = 1;
                int verts = pt.Faces[face].Verts;
                obj1.NSupp = 3;
                obj2.NSupp = 3;
                for (int i = 0; i < 3; i++)
                    CopySupport(pt, obj1, obj2, i, Vert(verts, i));
            } else {
                status.Nx = 0;
                status.Dist[0] = 0;
            }
            return face;
        }

        // ------------------------------------------------------------------------------------------ the multi-contact pass

        /// <summary>
        ///     More contacts from GJK's or EPA's support vertices (<c>mjc_multicontact</c>, MuJoCo 4), for a capsule (geom 1)
        ///     against a cylinder (geom 2): when the capsule's edge lies in a cylinder disk facing the contact (within
        ///     <c>mjEDGE_TOL</c>), the edge is clipped against the disk's 16-gon and each kept point becomes a witness pair,
        ///     inflated by the cylinder's half margin; otherwise the one contact stays.
        /// </summary>
        /// <param name="obj1">Geom 1 (the capsule).</param>
        /// <param name="obj2">Geom 2 (the cylinder).</param>
        /// <param name="x1">The witness points on geom 1, the first one in; rewritten.</param>
        /// <param name="x2">The witness points on geom 2, likewise.</param>
        /// <param name="dist">The distances, rewritten.</param>
        /// <param name="nconmax">The contacts wanted.</param>
        /// <returns>The witness pairs: 1, or the clipped points.</returns>
        /// <exception cref="NotSupportedException">The pair is not a capsule against a cylinder, or a face-face case.</exception>
        private static int Multicontact(CcdObject obj1, CcdObject obj2, double[] x1, double[] x2, double[] dist, int nconmax) {
            if (obj1.NSupp < 1 || obj2.NSupp < 1)
                return 1;
            if (obj1.GeomType != 3 || obj2.GeomType != 5)
                throw new NotSupportedException($"the convex collider: multi-contact for geom types {obj1.GeomType} and {obj2.GeomType}");
            Span<int> tri1i = stackalloc int[3], tri2i = stackalloc int[3];
            double[] tri1 = new double[9], tri2 = new double[9];
            int s11 = obj1.NSupp > 1 ? 1 : 0, s12 = obj1.NSupp > 2 ? 2 : 0;
            tri1i[0] = obj1.SuppInd[0];
            tri1i[1] = obj1.SuppInd[s11];
            tri1i[2] = obj1.SuppInd[s12];
            Array.Copy(obj1.SuppVert, 0, tri1, 0, 3);
            Array.Copy(obj1.SuppVert, 3 * s11, tri1, 3, 3);
            Array.Copy(obj1.SuppVert, 3 * s12, tri1, 6, 3);
            int nface1 = SimplexDim(tri1i, tri1);
            int s21 = obj2.NSupp > 1 ? 1 : 0, s22 = obj2.NSupp > 2 ? 2 : 0;
            tri2i[0] = obj2.SuppInd[0];
            tri2i[1] = obj2.SuppInd[s21];
            tri2i[2] = obj2.SuppInd[s22];
            Array.Copy(obj2.SuppVert, 0, tri2, 0, 3);
            Array.Copy(obj2.SuppVert, 3 * s21, tri2, 3, 3);
            Array.Copy(obj2.SuppVert, 3 * s22, tri2, 6, 3);
            int nface2 = SimplexDim(tri2i, tri2);
            Span<double> dir = stackalloc double[3], dirNeg = stackalloc double[3];
            for (int k = 0; k < 3; k++) {
                dir[k] = x2[k] - x1[k];
                dirNeg[k] = x1[k] - x2[k];
            }
            // face normals: a capsule has none; a cylinder whose support vertices share a disk has that disk's
            Span<double> n2 = stackalloc double[9];
            Span<int> idx2 = stackalloc int[3];
            int nnorms1 = 0;
            int nnorms2 = CylinderNormals(n2, idx2, nface2, obj2, tri2i);
            Span<int> res = stackalloc int[2];
            bool edgecon1 = false;
            // no aligned faces (the capsule has none); the capsule's edge against the cylinder's disk, as MuJoCo orders it
            bool swapOrder = (nface2 < nface1 || nnorms2 == 0) && nface2 < 3 && nnorms1 != 0;
            double[] edge = new double[3], endverts = new double[3];
            for (int k = 0; k < 2; k++) {
                int which = swapOrder ? 2 - k : 1 + k;
                if (which == 1 && nface1 < 3 && nnorms2 != 0) {
                    int nedge = CapsuleEdgeNormals(edge, endverts, obj1, tri1, x1);
                    if (AlignedFaceEdge(res, edge, nedge, n2, nnorms2, dir)) {
                        edgecon1 = true;
                        break;
                    }
                }
                // which == 2 needs geom 1's face normals, which a capsule does not have
            }
            if (!edgecon1)
                return 1;
            int i = res[0], j = res[1];
            double[] face1 = new double[6];
            Array.Copy(tri1, 0, face1, 0, 3);
            Array.Copy(endverts, 3 * i, face1, 3, 3);
            nface1 = 2;
            double[] face2 = new double[48];
            nface2 = CylinderFace(face2, obj2, idx2[j]);
            // the capsule's margin is already in its witness points; the cylinder's half margin is not
            double margin1 = 0, margin2 = 0.5 * obj2.Margin;
            double margin = margin1 + margin2;
            double[] witDir = new double[3], normal = new double[3];
            for (int k = 0; k < 3; k++)
                witDir[k] = -1.0 * n2[3 * j + k];
            witDir.CopyTo(normal, 0);
            int nx = PolygonClip(x2, x1, dist, nconmax, face2, nface2, face1, nface1, n2.Slice(3 * j, 3), witDir, margin);
            if (nx > 0) {
                Inflate(nx, x1, x2, dist, margin1, margin2, normal);
                return nx;
            }
            return 1;
        }

        /// <summary>
        ///     The dimension of a simplex from its vertex indices (<c>simplexDim</c>): 1 when all share one index, 2 for two
        ///     (the vertices reordered so the two come first), 3 otherwise.
        /// </summary>
        /// <param name="vi">The indices, reordered in place.</param>
        /// <param name="v">The vertices, 3 values each, reordered with them.</param>
        /// <returns>1, 2 or 3.</returns>
        private static int SimplexDim(Span<int> vi, double[] v) {
            if (vi[0] != vi[1])
                return vi[2] == vi[0] || vi[2] == vi[1] ? 2 : 3;
            if (vi[0] != vi[2]) {
                vi[1] = vi[2];
                Array.Copy(v, 6, v, 3, 3);
                return 2;
            }
            return 1;
        }

        /// <summary>
        ///     A cylinder's face normal from its support vertices (<c>cylinderNormals</c>): when they all lie on one disk, that
        ///     disk's outward normal (the axis, or its negation for the bottom); none otherwise.
        /// </summary>
        /// <param name="res">Receives the normal.</param>
        /// <param name="resind">Receives the disk's index (0 top, 1 bottom).</param>
        /// <param name="dim">The support simplex's dimension.</param>
        /// <param name="obj">The cylinder.</param>
        /// <param name="vi">The support vertices' indices.</param>
        /// <returns>1 with a normal, else 0.</returns>
        private static int CylinderNormals(Span<double> res, Span<int> resind, int dim, CcdObject obj, ReadOnlySpan<int> vi) {
            if (dim == 1) {
                GlobalCoord(res, obj.Mat, null, 0, 0, vi[0] != 0 ? -1.0 : 1.0);
                resind[0] = vi[0];
                return 1;
            }
            return 0;
        }

        /// <summary>
        ///     A capsule's edge from a witness point (<c>capsuleEdgeNormals</c>): the axis as the edge's direction, and the
        ///     segment through the point's projection, from <c>-pr - half-length</c> to <c>-pr + half-length</c> along it.
        /// </summary>
        /// <param name="res">Receives the axis.</param>
        /// <param name="endverts">Receives the segment's far end.</param>
        /// <param name="obj">The capsule.</param>
        /// <param name="v">Receives the segment's near end in its first three values (the support triangle's first vertex).</param>
        /// <param name="w">The witness point.</param>
        /// <returns>1 (one edge).</returns>
        private static int CapsuleEdgeNormals(double[] res, double[] endverts, CcdObject obj, double[] v, ReadOnlySpan<double> w) {
            res[0] = obj.Mat[2];
            res[1] = obj.Mat[5];
            res[2] = obj.Mat[8];
            Span<double> tmp = stackalloc double[3] { w[0] - obj.Pos[0], w[1] - obj.Pos[1], w[2] - obj.Pos[2] };
            double pr = Dot3(tmp, res);
            double length = obj.Size[1];
            for (int k = 0; k < 3; k++)
                v[k] = w[k] + (-pr - length) * res[k];
            for (int k = 0; k < 3; k++)
                endverts[k] = w[k] + (-pr + length) * res[k];
            return 1;
        }

        /// <summary>
        ///     An edge and a face normal perpendicular within <c>mjEDGE_TOL</c>, for a face facing the contact direction
        ///     (<c>alignedFaceEdge</c>).
        /// </summary>
        /// <param name="res">Receives the edge's and the face's indices.</param>
        /// <param name="edge">The edge directions, 3 values each.</param>
        /// <param name="nedge">Their count.</param>
        /// <param name="face">The face normals, 3 values each.</param>
        /// <param name="nface">Their count.</param>
        /// <param name="dir">The contact direction.</param>
        /// <returns>Whether a pair was found (the first in face, then edge, order).</returns>
        private static bool AlignedFaceEdge(Span<int> res, ReadOnlySpan<double> edge, int nedge, ReadOnlySpan<double> face, int nface,
                                            ReadOnlySpan<double> dir) {
            for (int i = 0; i < nface; i++) {
                if (Dot3(face.Slice(3 * i, 3), dir) <= Kernels.MinVal)
                    continue;
                for (int j = 0; j < nedge; j++) {
                    if (Math.Abs(Dot3(edge.Slice(3 * j, 3), face.Slice(3 * i, 3))) < EdgeTol) {
                        res[0] = j;
                        res[1] = i;
                        return true;
                    }
                }
            }
            return false;
        }

        /// <summary>The cosines of a 16-gon's angles, as MuJoCo's table writes them.</summary>
        private static readonly double[] Cos16 = [
            1.000000000000000, 0.923879532511287, 0.707106781186548, 0.382683432365090,
            0.000000000000000, -0.382683432365090, -0.707106781186547, -0.923879532511287,
            -1.000000000000000, -0.923879532511287, -0.707106781186548, -0.382683432365090,
            0.000000000000000, 0.382683432365090, 0.707106781186547, 0.923879532511287,
        ];

        /// <summary>The sines of a 16-gon's angles, as MuJoCo's table writes them.</summary>
        private static readonly double[] Sin16 = [
            0.000000000000000, 0.382683432365090, 0.707106781186547, 0.923879532511287,
            1.000000000000000, 0.923879532511287, 0.707106781186548, 0.382683432365090,
            0.000000000000000, -0.382683432365090, -0.707106781186547, -0.923879532511287,
            -1.000000000000000, -0.923879532511287, -0.707106781186548, -0.382683432365090,
        ];

        /// <summary>
        ///     A cylinder disk as a 16-gon (<c>cylinderFace</c>): the top (<paramref name="idx"/> 0) or bottom disk, wound so
        ///     its normal points out.
        /// </summary>
        /// <param name="res">Receives the 16 vertices, 3 values each.</param>
        /// <param name="obj">The cylinder.</param>
        /// <param name="idx">The disk: 0 top, otherwise bottom.</param>
        /// <returns>16.</returns>
        private static int CylinderFace(double[] res, CcdObject obj, int idx) {
            double sgn = idx != 0 ? -1.0 : 1.0;
            for (int i = 0; i < 16; i++) {
                double x = Cos16[i] * obj.Size[0];
                double y = -Sin16[i] * obj.Size[0] * sgn;
                GlobalCoord(res.AsSpan(3 * i, 3), obj.Mat, obj.Pos, x, y, sgn * obj.Size[1]);
            }
            return 16;
        }

        /// <summary><c>globalcoord</c>: <c>mat*(l1, l2, l3) + pos</c>, the position added after the rotation, when given.</summary>
        /// <param name="res">Receives the point.</param>
        /// <param name="mat">The rotation, row-major.</param>
        /// <param name="pos">The position, or <see langword="null"/>.</param>
        /// <param name="l1">The local x.</param>
        /// <param name="l2">The local y.</param>
        /// <param name="l3">The local z.</param>
        private static void GlobalCoord(Span<double> res, double[] mat, double[]? pos, double l1, double l2, double l3) {
            res[0] = mat[0] * l1 + mat[1] * l2 + mat[2] * l3;
            res[1] = mat[3] * l1 + mat[4] * l2 + mat[5] * l3;
            res[2] = mat[6] * l1 + mat[7] * l2 + mat[8] * l3;
            if (pos != null) {
                res[0] += pos[0];
                res[1] += pos[1];
                res[2] += pos[2];
            }
        }

        /// <summary>
        ///     The normal of the plane through a face edge, perpendicular to the face (<c>planeNormal</c>), normalized, and the
        ///     plane's offset along it.
        /// </summary>
        /// <param name="res">Receives the unit normal.</param>
        /// <param name="v1">The edge's start.</param>
        /// <param name="v2">The edge's end.</param>
        /// <param name="n">The face normal.</param>
        /// <returns><c>res·v1</c>.</returns>
        private static double PlaneNormal(Span<double> res, ReadOnlySpan<double> v1, ReadOnlySpan<double> v2, ReadOnlySpan<double> n) {
            Span<double> v3 = stackalloc double[3], d1 = stackalloc double[3], d2 = stackalloc double[3];
            for (int k = 0; k < 3; k++) {
                v3[k] = v1[k] + n[k];
                d1[k] = v2[k] - v1[k];
            }
            for (int k = 0; k < 3; k++)
                d2[k] = v3[k] - v1[k];
            Spatial.Cross(res, d1, d2);
            Spatial.Normalize3(res);
            return Dot3(res, v1);
        }

        /// <summary>Whether a point lies on the normal's side of a plane, within <c>mjMINVAL</c> (<c>halfspace</c>).</summary>
        /// <param name="a">A point of the plane.</param>
        /// <param name="n">The plane's normal.</param>
        /// <param name="p">The point.</param>
        /// <returns>Whether <c>(p - a)·n &gt; -mjMINVAL</c>.</returns>
        private static bool Halfspace(ReadOnlySpan<double> a, ReadOnlySpan<double> n, ReadOnlySpan<double> p) {
            Span<double> diff = stackalloc double[3] { p[0] - a[0], p[1] - a[1], p[2] - a[2] };
            return Dot3(diff, n) > -Kernels.MinVal;
        }

        /// <summary>
        ///     The witness pair of a clipped vertex (<c>witnessOnFace</c>): its projection on the face, and that point moved
        ///     back along the contact direction by its distance.
        /// </summary>
        /// <param name="w1">Receives the point on the face.</param>
        /// <param name="w2">Receives the point on the other geom.</param>
        /// <param name="v">The clipped vertex.</param>
        /// <param name="p">A point of the face.</param>
        /// <param name="n">The face normal.</param>
        /// <param name="dir">The contact direction.</param>
        /// <returns>The signed distance along the direction.</returns>
        private static double WitnessOnFace(Span<double> w1, Span<double> w2, ReadOnlySpan<double> v, ReadOnlySpan<double> p,
                                            ReadOnlySpan<double> n, ReadOnlySpan<double> dir) {
            Span<double> d = stackalloc double[3] { v[0] - p[0], v[1] - p[1], v[2] - p[2] };
            double s = -Dot3(d, n);
            for (int k = 0; k < 3; k++)
                w1[k] = v[k] + s * n[k];
            for (int k = 0; k < 3; k++)
                d[k] = w1[k] - v[k];
            double dist = Dot3(d, dir);
            for (int k = 0; k < 3; k++)
                w2[k] = w1[k] + -dist * dir[k];
            return dist;
        }

        /// <summary>
        ///     The area of a quadrilateral (<c>area4</c>): half the norm of its diagonals' cross product.
        /// </summary>
        /// <param name="hull">The hull's points, 3 values each.</param>
        /// <param name="a">The first corner.</param>
        /// <param name="b">The second.</param>
        /// <param name="c">The third.</param>
        /// <param name="d">The fourth.</param>
        /// <returns>The area.</returns>
        private static double Area4(ReadOnlySpan<double> hull, int a, int b, int c, int d) {
            Span<double> ca = stackalloc double[3], db = stackalloc double[3], cross = stackalloc double[3];
            for (int k = 0; k < 3; k++) {
                ca[k] = hull[3 * a + k] - hull[3 * c + k];
                db[k] = hull[3 * b + k] - hull[3 * d + k];
            }
            Spatial.Cross(cross, ca, db);
            return 0.5 * Norm3(cross);
        }

        /// <summary>
        ///     The largest-area quadrilateral of a convex hull (<c>hull4</c>), by MuJoCo's rotating search.
        /// </summary>
        /// <param name="res">Receives the four corners' indices.</param>
        /// <param name="hull">The hull's points, 3 values each.</param>
        /// <param name="nhull">The hull's point count.</param>
        private static void Hull4(Span<int> res, ReadOnlySpan<double> hull, int nhull) {
            int a = 0, b = 1, c = 2, d = 3;
            res[0] = 0;
            res[1] = 1;
            res[2] = 2;
            res[3] = 3;
            double m = Area4(hull, a, b, c, d), mNext;
            for (; a < nhull; a++) {
                while (true) {
                    int dNext = (d + 1) % nhull;
                    mNext = Area4(hull, a, b, c, dNext);
                    if (mNext <= m)
                        break;
                    d = dNext;
                    m = mNext;
                    res[0] = a;
                    res[1] = b;
                    res[2] = c;
                    res[3] = d;
                    while (true) {
                        int cNext = (c + 1) % nhull;
                        mNext = Area4(hull, a, b, cNext, d);
                        if (mNext <= m)
                            break;
                        c = cNext;
                        m = mNext;
                        res[0] = a;
                        res[1] = b;
                        res[2] = c;
                        res[3] = d;
                    }
                    while (true) {
                        int bNext = (b + 1) % nhull;
                        mNext = Area4(hull, a, bNext, c, d);
                        if (mNext <= m)
                            break;
                        b = bNext;
                        m = mNext;
                        res[0] = a;
                        res[1] = b;
                        res[2] = c;
                        res[3] = d;
                    }
                }
                if (b == a) {
                    b = (b + 1) % nhull;
                    if (c == b) {
                        c = (c + 1) % nhull;
                        if (d == c)
                            d = (d + 1) % nhull;
                    }
                }
            }
        }

        /// <summary>
        ///     Clips a polygon against a face's edges, one at a time (<c>polygonClip</c>), keeps the points within the margin of
        ///     the face, and turns them into witness pairs: four by area when more remain than four and fewer contacts are
        ///     wanted than five, the two farthest apart for an edge, else up to <paramref name="nconmax"/>.
        /// </summary>
        /// <param name="x1">Receives the points on the clipping face.</param>
        /// <param name="x2">Receives the points on the other geom.</param>
        /// <param name="dist">Receives the distances.</param>
        /// <param name="nconmax">The contacts wanted.</param>
        /// <param name="face1">The clipping face, 3 values per vertex.</param>
        /// <param name="nface1">Its vertex count (at least 3).</param>
        /// <param name="face2">The clipped polygon (an edge: 2 vertices).</param>
        /// <param name="nface2">Its vertex count.</param>
        /// <param name="n">The clipping face's normal.</param>
        /// <param name="dir">The contact direction.</param>
        /// <param name="margin">The distance from the face within which points are kept.</param>
        /// <returns>The witness pairs written (0 when nothing survives).</returns>
        private static int PolygonClip(double[] x1, double[] x2, double[] dist, int nconmax, double[] face1, int nface1, double[] face2,
                                       int nface2, ReadOnlySpan<double> n, ReadOnlySpan<double> dir, double margin) {
            if (nface1 != 0 && nface1 < 3)
                return 0;
            int cap = 2 * (nface1 + nface2) + 8;
            double[] polygon = new double[3 * cap], clipped = new double[3 * cap];
            double[] pn = new double[3 * nface1], pd = new double[nface1];
            for (int i = 0; i < nface1 - 1; i++)
                pd[i] = PlaneNormal(pn.AsSpan(3 * i, 3), face1.AsSpan(3 * i, 3), face1.AsSpan(3 * i + 3, 3), n);
            pd[nface1 - 1] = PlaneNormal(pn.AsSpan(3 * (nface1 - 1), 3), face1.AsSpan(3 * (nface1 - 1), 3), face1.AsSpan(0, 3), n);
            int npolygon = nface2, nclipped = 0;
            Array.Copy(face2, polygon, 3 * nface2);
            Span<double> pq = stackalloc double[3];
            for (int e = 0; e < 3 * nface1; e += 3) {
                for (int i = 0; i < npolygon; i++) {
                    int p = 3 * i, q = i < npolygon - 1 ? 3 * (i + 1) : 0;
                    for (int k = 0; k < 3; k++)
                        pq[k] = polygon[q + k] - polygon[p + k];
                    bool inside1 = Halfspace(face1.AsSpan(e, 3), pn.AsSpan(e, 3), polygon.AsSpan(p, 3));
                    bool inside2 = Halfspace(face1.AsSpan(e, 3), pn.AsSpan(e, 3), polygon.AsSpan(q, 3));
                    if (!inside1 && !inside2)
                        continue;
                    if (inside1 && inside2) {
                        Array.Copy(polygon, q, clipped, 3 * nclipped++, 3);
                        continue;
                    }
                    // the edge crosses the clipping plane: add the crossing
                    double distP = (polygon[p] - face1[e]) * pn[e] + (polygon[p + 1] - face1[e + 1]) * pn[e + 1]
                                   + (polygon[p + 2] - face1[e + 2]) * pn[e + 2];
                    double distQ = (polygon[q] - face1[e]) * pn[e] + (polygon[q + 1] - face1[e + 1]) * pn[e + 1]
                                   + (polygon[q + 2] - face1[e + 2]) * pn[e + 2];
                    double denom = distP - distQ;
                    if (denom != 0.0) {
                        double t = distP / denom;
                        if (t >= 0.0 && t <= 1.0) {
                            int o = 3 * nclipped++;
                            for (int k = 0; k < 3; k++)
                                clipped[o + k] = polygon[p + k] + t * pq[k];
                        }
                    }
                    if (inside2)
                        Array.Copy(polygon, q, clipped, 3 * nclipped++, 3);
                }
                (polygon, clipped) = (clipped, polygon);
                npolygon = nclipped;
                nclipped = 0;
            }
            // keep the points within the margin of the face
            int count = npolygon;
            npolygon = 0;
            for (int i = 0; i < count; i++) {
                double d = (polygon[3 * i] - face1[0]) * n[0] + (polygon[3 * i + 1] - face1[1]) * n[1] + (polygon[3 * i + 2] - face1[2]) * n[2];
                if (d <= margin) {
                    if (npolygon != i)
                        Array.Copy(polygon, 3 * i, polygon, 3 * npolygon, 3);
                    npolygon++;
                }
            }
            if (npolygon < 1)
                return 0;
            if (nconmax < 5 && npolygon > 4) {
                Span<int> idx = stackalloc int[4];
                Hull4(idx, polygon, npolygon);
                for (int i = 0; i < 4; i++)
                    dist[i] = WitnessOnFace(x1.AsSpan(3 * i, 3), x2.AsSpan(3 * i, 3), polygon.AsSpan(3 * idx[i], 3), face1, n, dir);
                return 4;
            }
            // an edge: the two points farthest apart (the first pair on ties)
            if (nface2 == 2 && npolygon > 2) {
                int best1 = 0, best2 = 1;
                double dd = 0;
                for (int i = 0; i < npolygon; i++) {
                    for (int j = i + 1; j < npolygon; j++) {
                        double e0 = polygon[3 * j] - polygon[3 * i], e1 = polygon[3 * j + 1] - polygon[3 * i + 1], e2 = polygon[3 * j + 2] - polygon[3 * i + 2];
                        double d2 = e0 * e0 + e1 * e1 + e2 * e2;
                        if (d2 > dd) {
                            dd = d2;
                            best1 = i;
                            best2 = j;
                        }
                    }
                }
                double d0 = WitnessOnFace(x1.AsSpan(0, 3), x2.AsSpan(0, 3), polygon.AsSpan(3 * best1, 3), face1, n, dir);
                double d1 = WitnessOnFace(x1.AsSpan(3, 3), x2.AsSpan(3, 3), polygon.AsSpan(3 * best2, 3), face1, n, dir);
                dist[0] = d0;
                dist[1] = d1;
                return 2;
            }
            npolygon = npolygon < nconmax ? npolygon : nconmax;
            for (int i = 0; i < npolygon; i++)
                dist[i] = WitnessOnFace(x1.AsSpan(3 * i, 3), x2.AsSpan(3 * i, 3), polygon.AsSpan(3 * i, 3), face1, n, dir);
            return npolygon;
        }
    }

    /// <summary>An EPA face (<c>Face</c>): packed vertices, neighbours per edge, the origin's projection, its map slot.</summary>
    internal struct CcdFace {
        /// <summary>The three vertex indices, 10 bits each.</summary>
        public int Verts;
        /// <summary>The neighbour across edge (v1, v2).</summary>
        public int Adj0;
        /// <summary>The neighbour across edge (v2, v3).</summary>
        public int Adj1;
        /// <summary>The neighbour across edge (v3, v1).</summary>
        public int Adj2;
        /// <summary>The origin's projection on the face's plane, x.</summary>
        public double V0;
        /// <summary>The projection, y.</summary>
        public double V1;
        /// <summary>The projection, z.</summary>
        public double V2;
        /// <summary>The projection's squared norm.</summary>
        public double Dist2;
        /// <summary>The face's slot in the candidate map; -1 not in it, -2 deleted.</summary>
        public int Index;
    }

    /// <summary>
    ///     The expanding polytope (<c>Polytope</c>): vertices as rows of 9 values with their two vertex indices, faces, the
    ///     map of candidate faces and the horizon of the current expansion, sized as MuJoCo sizes its buffer for an iteration
    ///     budget.
    /// </summary>
    internal sealed class CcdPolytope {
        /// <summary>Vertex rows (difference point, geom 1 point, geom 2 point).</summary>
        public double[] Verts = [];
        /// <summary>Each vertex's index on geom 1 (<c>index1</c>).</summary>
        public int[] Index1 = [];
        /// <summary>Each vertex's index on geom 2 (<c>index2</c>).</summary>
        public int[] Index2 = [];
        /// <summary>The vertex count.</summary>
        public int Nverts;
        /// <summary>The faces.</summary>
        public CcdFace[] Faces = [];
        /// <summary>The face count.</summary>
        public int Nfaces;
        /// <summary>The face capacity (<c>6*iterations</c>).</summary>
        public int MaxFaces;
        /// <summary>The initial polytope's centre, against which faces are turned outward.</summary>
        public readonly double[] Center = new double[3];
        /// <summary>The candidate faces (indices into <see cref="Faces"/>).</summary>
        public int[] Map = [];
        /// <summary>The candidate count.</summary>
        public int Nmap;
        /// <summary>The horizon's faces.</summary>
        public int[] HorizonIndices = [];
        /// <summary>The horizon's edges, one per face.</summary>
        public int[] HorizonEdges = [];
        /// <summary>The horizon's edge count (it keeps counting past <see cref="MaxEdges"/>).</summary>
        public int Nedges;
        /// <summary>The horizon's capacity (<c>24 + max(0, iterations - 35)</c>).</summary>
        public int MaxEdges;
        /// <summary>The offset of the vertex the horizon is seen from.</summary>
        public int HorizonW;

        /// <summary>
        ///     Empties the polytope and sizes it for an iteration budget (vertices <c>5+N</c>, faces <c>6N</c>, horizon edges
        ///     <c>24 + max(0, N - 35)</c>), reusing the arrays when they are large enough.
        /// </summary>
        /// <param name="iterations">The budget.</param>
        public void Reset(int iterations) {
            if (Faces.Length < 6 * iterations) {
                Verts = new double[(5 + iterations) * Ccd.VertexSize];
                Index1 = new int[5 + iterations];
                Index2 = new int[5 + iterations];
                Faces = new CcdFace[6 * iterations];
                Map = new int[6 * iterations];
            }
            MaxFaces = 6 * iterations;
            MaxEdges = 24 + (iterations > 35 ? iterations - 35 : 0);
            if (HorizonIndices.Length < MaxEdges) {
                HorizonIndices = new int[MaxEdges];
                HorizonEdges = new int[MaxEdges];
            }
            Nverts = Nfaces = Nmap = Nedges = 0;
        }
    }
}
