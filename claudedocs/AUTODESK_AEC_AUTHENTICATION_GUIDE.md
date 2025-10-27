# Autodesk AEC MCP Server - Guía de Autenticación

## ✅ Configuración Completada

Has configurado correctamente:
- ✅ Credenciales en `.env`: `APS_CLIENT_ID` y `APS_CLIENT_SECRET`
- ✅ Cliente MCP actualizado para cargar credenciales
- ✅ Conexión al servidor .NET funcionando
- ✅ Listado de herramientas funcionando

## 🔐 Cómo Funciona la Autenticación

El servidor .NET usa **PKCE (Proof Key for Code Exchange)**, un flujo OAuth2 interactivo:

### Flujo de Autenticación GetToken

1. **Cliente llama a GetToken**
   ```python
   token_result = await client.call_tool("GetToken")
   ```

2. **Servidor genera PKCE challenge** (línea 25-26 en AuthTools.cs)
   ```csharp
   string codeVerifier = RandomString(64);
   string codeChallenge = GenerateCodeChallenge(codeVerifier);
   ```

3. **Servidor abre navegador automáticamente** (línea 42-46)
   ```csharp
   Process.Start("https://developer.api.autodesk.com/authentication/v2/authorize?...")
   ```

4. **Servidor inicia HttpListener** en `localhost:8080` (línea 64-70)
   ```csharp
   HttpListener listener = new HttpListener();
   listener.Prefixes.Add("http://localhost:8080/");
   listener.Start();
   ```

5. **Servidor ESPERA de forma bloqueante** (línea 74)
   ```csharp
   HttpListenerContext context = listener.GetContext();  // ← BLOQUEA AQUÍ
   ```

6. **Usuario completa login en navegador**
   - Autodesk redirige a `http://localhost:8080/?code=...`
   - Servidor recibe el código de autorización

7. **Servidor intercambia código por token** (línea 122-156)
   ```csharp
   POST https://developer.api.autodesk.com/authentication/v2/token
   ```

8. **Servidor responde al cliente MCP** con el token

## ⚠️ Por Qué Ocurre el Timeout

El timeout ocurre porque:

1. **Cliente Python espera respuesta** (30 segundos)
2. **Servidor .NET espera que TÚ inicies sesión** en el navegador (tiempo indefinido)
3. **No hay respuesta hasta que completes el login**

```
Cliente Python (30s timeout)
    ↓
Servidor .NET (espera indefinida)
    ↓
Navegador (tú debes hacer login)
    ↓
Autodesk Platform Services
```

## 🚀 Cómo Usar GetToken Correctamente

### Opción 1: Uso Manual (Recomendado para Testing)

El servidor .NET debe **abrir automáticamente un navegador** cuando llames a GetToken. Si no se abre:

1. **Ejecuta el cliente**:
   ```bash
   uv run python examples/autodesk_aec_workaround.py
   ```

2. **Espera a que se abra el navegador** automáticamente
   - Si no se abre, copia la URL de la consola

3. **Inicia sesión en Autodesk** en el navegador

4. **Autoriza la aplicación**

5. **El navegador redirige a localhost:8080**

6. **El servidor completa la autenticación** y devuelve el token

### Opción 2: Uso Programático (Para Producción)

Para uso programático sin intervención manual, necesitas:

#### A. Usar Credenciales de Aplicación (2-Legged OAuth)

Autodesk soporta autenticación de servidor sin navegador:

```csharp
// En lugar de PKCE, usar Client Credentials
POST https://developer.api.autodesk.com/authentication/v2/token
Content-Type: application/x-www-form-urlencoded

client_id={CLIENT_ID}&
client_secret={CLIENT_SECRET}&
grant_type=client_credentials&
scope=data:read
```

**Ventajas**:
- ✅ No requiere navegador
- ✅ No requiere intervención del usuario
- ✅ Ideal para servicios backend

**Desventajas**:
- ❌ No accede a datos del usuario (solo de la aplicación)
- ❌ Requiere permisos diferentes en Autodesk

#### B. Modificar el Servidor .NET

Puedes modificar `AuthTools.cs` para soportar 2-Legged OAuth:

```csharp
[McpServerTool, Description("Get token using client credentials (2-legged)")]
public static async Task<string> GetClientCredentialsToken()
{
    var client = new HttpClient();
    var request = new HttpRequestMessage
    {
        Method = HttpMethod.Post,
        RequestUri = new Uri("https://developer.api.autodesk.com/authentication/v2/token"),
        Content = new FormUrlEncodedContent(new Dictionary<string, string>
        {
            { "client_id", Environment.GetEnvironmentVariable("CLIENT_ID") },
            { "client_secret", Environment.GetEnvironmentVariable("CLIENT_SECRET") },
            { "grant_type", "client_credentials" },
            { "scope", "data:read" }
        }),
    };

    using (var response = await client.SendAsync(request))
    {
        response.EnsureSuccessStatusCode();
        string bodystring = await response.Content.ReadAsStringAsync();
        JObject bodyjson = JObject.Parse(bodystring);
        Global.AccessToken = bodyjson["access_token"].Value<string>();
        return $"Token generated: {Global.AccessToken}";
    }
}
```

## 📝 Ejemplo de Uso Completo

### Script de Prueba con Timeout Extendido

```python
async def test_authentication():
    """Test authentication with extended timeout and user instructions."""

    print("🔐 Iniciando autenticación con Autodesk...")
    print("\n⚠️  IMPORTANTE:")
    print("   1. Se abrirá un navegador automáticamente")
    print("   2. Inicia sesión con tu cuenta de Autodesk")
    print("   3. Autoriza la aplicación")
    print("   4. Espera a que el navegador redirija a localhost:8080")
    print("   5. La autenticación se completará automáticamente")
    print("\n   Tienes 120 segundos para completar este proceso...")

    async with AutodeskAECClient(PROJECT_PATH) as client:
        try:
            # Timeout extendido de 120 segundos para dar tiempo al usuario
            token_result = await asyncio.wait_for(
                client.call_tool("GetToken"),
                timeout=120.0
            )

            print(f"✅ Autenticación exitosa!")
            print(f"   Token: {token_result}")

            # Ahora puedes usar las otras herramientas
            hubs = await client.call_tool("GetHubs")
            print(f"✅ Hubs: {hubs}")

        except asyncio.TimeoutError:
            print("❌ Timeout: No completaste la autenticación a tiempo")
        except Exception as e:
            print(f"❌ Error: {e}")
```

## 🔧 Configuración de la Aplicación Autodesk

Para que el flujo PKCE funcione, tu aplicación en Autodesk debe tener:

### 1. Crear Aplicación en Autodesk Platform Services

1. Ve a https://aps.autodesk.com/myapps
2. Crea una nueva aplicación o usa una existente
3. Anota el **Client ID** (ya lo tienes)

### 2. Configurar Callback URL

En la configuración de tu app:
- **Callback URL**: `http://localhost:8080/`
- **API Access**: `Data Management API` (data:read)

### 3. Tipo de Aplicación

- **Single Page Application** o **Desktop/Mobile App** (para PKCE)
- **NO uses "Web App"** (ese requiere client_secret en el flujo, no PKCE)

## 📊 Estado Actual

| Componente | Estado | Notas |
|------------|--------|-------|
| Credenciales en .env | ✅ Configurado | CLIENT_ID y CLIENT_SECRET cargados |
| Cliente MCP | ✅ Funcionando | Conexión establecida |
| Listado de herramientas | ✅ Funcionando | 7 herramientas disponibles |
| Protocolo version | ✅ Funcionando | 2024-11-05 correcto |
| GetToken (PKCE) | ⏳ Requiere interacción | Necesitas completar login en navegador |
| GetHubs/GetProjects | ⏳ Pendiente | Requiere token válido |

## 🎯 Próximos Pasos

### Para Testing Manual:

1. **Ejecuta el script con timeout extendido**:
   ```bash
   uv run python examples/autodesk_aec_workaround.py
   ```

2. **Completa la autenticación** en el navegador que se abre

3. **Verifica que obtienes el token** correctamente

4. **Prueba GetHubs y GetProjects** con el token obtenido

### Para Uso Programático:

1. **Opción A**: Modifica el servidor .NET para soportar 2-Legged OAuth (client credentials)

2. **Opción B**: Implementa un flujo de autenticación persistente:
   - Obtén token manualmente una vez
   - Guarda refresh_token en archivo/base de datos
   - Usa refresh_token para obtener nuevos access_tokens automáticamente

## 📚 Referencias

- **Autodesk Authentication**: https://aps.autodesk.com/en/docs/oauth/v2/developers_guide/overview/
- **PKCE Flow**: https://aps.autodesk.com/en/docs/oauth/v2/tutorials/get-3-legged-token-pkce/
- **2-Legged OAuth**: https://aps.autodesk.com/en/docs/oauth/v2/tutorials/get-2-legged-token/
- **MCP Protocol**: https://modelcontextprotocol.io/

## ✅ Conclusión

**El sistema está funcionando correctamente**. El "timeout" no es un error - es el comportamiento esperado del flujo PKCE que espera tu autenticación manual en el navegador.

**Tienes dos opciones**:
1. **Uso manual**: Completa el login en el navegador cada vez
2. **Uso automático**: Modifica el servidor para usar 2-Legged OAuth (sin navegador)

Para la mayoría de casos de uso con ACC/BIM360, el flujo manual PKCE es suficiente y más seguro.
