# Guide: Development, Build & Test Automation

[Back to Documentation Index](../README.md) · [View PRD](../PRD.md)

---

## 1. Prerequisites & Toolchain

- **.NET SDK**: .NET 8.0 SDK / .NET 10.0 SDK.
- **Runtimes**: Windows Desktop Runtime (for WinForms / Avalonia test runners).
- **Submodules**: Ensure `refs/Gymnasium` submodule is initialized:
  ```powershell
  git submodule update --init --recursive
  ```

---

## 2. Build Commands

### 2.1 Build Unified Solution
```powershell
# Debug configuration
dotnet build Gym.NET.sln -c Debug

# Release configuration
dotnet build Gym.NET.sln -c Release
```

### 2.2 Build Specific Subprojects
```powershell
dotnet build src/Gym/Gym.csproj -c Release
dotnet build src/Gym.Environments/Gym.Environments.csproj -c Release
dotnet build src/Gym.Rendering.Avalonia/Gym.Rendering.Avalonia.csproj -c Release
dotnet build src/Gym.Rendering.WinForm/Gym.Rendering.WinForm.csproj -c Release
```

---

## 3. Automated Test Execution (MSTest)

### 3.1 Run Full Test Suite Across All Frameworks
```powershell
dotnet test tests/Gym.Tests/Gym.Tests.csproj -c Release
```

### 3.2 Target Specific Frameworks
```powershell
# Run tests targeting .NET 8.0
dotnet test tests/Gym.Tests/Gym.Tests.csproj -f net8.0-windows

# Run tests targeting .NET 10.0
dotnet test tests/Gym.Tests/Gym.Tests.csproj -f net10.0-windows
```

### 3.3 Filter by Test Categories
```powershell
# Run only Box space tests
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~BoxTest"

# Run only headless environment tests (safe for CI / Server Core)
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~NullEnv"

# Run Avalonia UI rendering tests
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~AvaloniaEnv"

# Run a single individual test
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~CartpoleEnvironment.Run_NullEnv"
```
