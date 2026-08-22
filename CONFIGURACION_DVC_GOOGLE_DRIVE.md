# Configuración inicial de DVC + Google Drive

Esta guía explica los pasos que cada integrante del equipo debe seguir **una sola vez** para poder hacer `dvc pull` (descargar datos) y `dvc push` (subir datos) desde el remoto en Google Drive.

## 1. Requisitos previos

- Tener el repositorio clonado localmente:
  ```bash
  git clone <url-del-repo>
  cd Microproyecto
  ```
- Tener un entorno virtual de Python activado :
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

## 2. Instalar DVC con soporte para Google Drive

```bash
pip install "dvc[gdrive]"
```


### 3. Configurar el acceso al remoto (Google Drive)

La ruta hacia el almacenamiento de datos ya viene versionada en el repositorio. Al hacer `git pull`, tendrás el remoto apuntado automáticamente. Puedes verificarlo con:

```bash
dvc remote default gdrive-remote
dvc remote list

```

*(Deberías ver: `gdrive-remote   gdrive://<ID_DE_CARPETA>`)*

Por seguridad, las credenciales de acceso no se suben a Git. Para poder descargar (`pull`) o subir (`push`) datos, debes configurar las claves proporcionadas por el equipo de forma local. Ejecuta los siguientes comandos en tu terminal:

```bash
dvc remote modify --local gdrive-remote gdrive_client_id "TU_CLIENT_ID"
dvc remote modify --local gdrive-remote gdrive_client_secret "TU_CLIENT_SECRET"

```

Esto guardará las llaves en un archivo oculto (`.dvc/config.local`) que está excluido del control de versiones, garantizando que el acceso se mantenga seguro y privado en tu computadora.

## 4. Estar configurado como "test user" en Google Cloud

Como el proyecto usa una app de OAuth propia (no verificada públicamente por Google), **cada persona del equipo debe estar agregada como usuario de prueba** (`Google Cloud Console → APIs & Services → OAuth consent screen → Audience → Test users`) para poder autenticarse. Sin este paso, van a ver un error de "This app is blocked" al intentar conectarse. **TODOS LOS MIEMBROS DEL EQUIPO ESTAN CONFIGURADOS COMO TEST USERS**


## 5. Ejecutar el primer `dvc pull`

Una vez agregado como test user, simplemente corre:

```bash
dvc pull
```

Esto va a:
1. Abrir el navegador automáticamente para iniciar sesión con tu cuenta de Google.
2. Pedir aprbación para el acceso a Google Drive (aparecerá el nombre de la app configurada para este propósito).
3. Descargar los datos versionados desde el remoto a tu carpeta local.

> DVC guarda el token de autenticación localmente después de la primera vez, así que **no será necesario volver a iniciar sesión** en cada `pull`/`push` posterior (a menos que el token expire o se borre el caché local de credenciales).

## 6. Verificar que todo esté sincronizado

```bash
dvc status -c
```

Si todo salió bien, debería decir algo como:
```
Cache and remote 'gdrive-remote' are in sync.
```

## Problemas comunes

| Problema | Causa probable | Solución |
|---|---|---|
| `This app is blocked` en el navegador | No estás agregado como test user en Google Cloud | Pide que te agreguen en el paso 4 |

| Error tipo `module 'lib' has no attribute 'GEN_EMAIL'` | Conflicto de versiones entre `pyOpenSSL` y `cryptography` | Ejecuta `pip install "pyopenssl==24.2.1" "cryptography<44"` |
| El navegador no abre automáticamente | Puede pasar en entornos remotos/sin interfaz gráfica | Copia el link que aparece en la terminal y ábrelo manualmente en tu navegador |

## Flujo del día a día (una vez configurado)

```bash
git pull        # trae los archivos .dvc actualizados
dvc pull        # descarga los datos reales desde Google Drive
```

Subir cambios en los datos:

```bash
dvc add <archivo>
git add <archivo>.dvc
git commit -m "mensaje descriptivo"
dvc push
git push
```
