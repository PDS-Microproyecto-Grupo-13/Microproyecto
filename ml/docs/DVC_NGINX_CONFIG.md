# Configuración y Uso de Clientes DVC

Guía operativa para conectar clientes DVC al servidor autogestionado definido en [`DVC_NGINX_SERVER.md`](DVC_NGINX_SERVER.md), utilizando dos canales independientes sobre el mismo almacenamiento:

- **`dvc-public`**: remoto HTTP de solo lectura para `dvc pull`, sin credenciales.
- **`dvc-write`**: remoto SSH de lectura/escritura para `dvc push`, autenticado mediante clave pública.

Este documento asume que el servidor ya se encuentra operativo con:

```text
HTTP público:  http://<DVC_HOST>:8080/dvc/
SSH privado:   ssh://dvcuser@<DVC_HOST>:2222/srv/dvc-storage
Storage:       /srv/dvc-storage
```

No se requiere instalar DVC en el servidor. DVC se ejecuta únicamente en las máquinas cliente.

---

## 1. Modelo de acceso

```text
                         /srv/dvc-storage
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 │                           │
           SSH :2222                    HTTP :8080
        autenticado por key             sin credenciales
                 │                           │
                 v                           v
          dvc-write                     dvc-public
          push (RW)                     pull (RO)
                 │                           │
                 └─────────────┬─────────────┘
                               │
                         clientes DVC
```

La regla operacional es:

```text
Descargar datos  -> dvc-public
Subir datos      -> dvc-write
```

El remoto HTTP se comparte con todos los usuarios y puede versionarse en Git.

El remoto SSH contiene configuración específica de cada escritor y se mantiene únicamente en `.dvc/config.local`.

---

## 2. Tipos de cliente

### Consumidor

Solo necesita:

```text
Git
DVC
acceso HTTP al puerto 8080
```

Puede ejecutar:

```bash
dvc pull
```

sin credenciales.

### Escritor

Además necesita:

```text
soporte SSH para DVC
clave privada autorizada por el servidor
acceso TCP al puerto 2222
```

Puede ejecutar:

```bash
dvc push -r dvc-write
```

---

## 3. Preparar el entorno DVC

Desde el repositorio:

```bash
cd Microproyecto/ml
```

Si se utiliza el entorno Python del módulo ML:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.lock.txt
```

Verificar:

```bash
dvc version
```

Para un cliente que realizará `push` mediante SSH, instalar además el soporte SSH de DVC si no está disponible:

```bash
pip install "dvc[ssh]"
```

---

## 4. Configurar el remoto público HTTP

Esta configuración pertenece al proyecto y debe quedar versionada en Git.

Desde `ml/`:

```bash
dvc remote add -d dvc-public \
  http://<DVC_HOST>:8080/dvc/
```

Verificar:

```bash
dvc remote list
```

Debe aparecer:

```text
dvc-public    http://<DVC_HOST>:8080/dvc/
```

La configuración resultante queda almacenada en:

```text
.dvc/config
```

Conceptualmente:

```ini
[core]
    remote = dvc-public

['remote "dvc-public"']
    url = http://<DVC_HOST>:8080/dvc/
```

Versionar el cambio:

```bash
git add .dvc/config
git commit -m "Configure public DVC remote"
```

A partir de este momento, cualquier clone que contenga esta configuración utilizará `dvc-public` como remoto predeterminado.

---

## 5. Configurar un cliente escritor

Esta sección se realiza solamente en máquinas autorizadas para subir objetos.

### 5.1 Verificar primero el acceso SSH

Desde la máquina cliente:

```bash
ssh \
  -i ~/.ssh/salarypredict/dvc_server_key \
  -p 2222 \
  dvcuser@<DVC_HOST>
```

La conexión debe iniciar sesión como:

```text
dvcuser
```

Salir:

```bash
exit
```

Si esta prueba falla, no continuar con la configuración DVC hasta resolver el acceso SSH.

---

## 6. Agregar el remoto privado de escritura

Desde `ml/`:

```bash
dvc remote add --local dvc-write \
  ssh://dvcuser@<DVC_HOST>:2222/srv/dvc-storage
```

Indicar la clave privada:

```bash
dvc remote modify --local dvc-write \
  keyfile ~/.ssh/salarypredict/dvc_server_key
```

Verificar:

```bash
dvc remote list
```

Debe aparecer una configuración equivalente a:

```text
dvc-public    http://<DVC_HOST>:8080/dvc/
dvc-write     ssh://dvcuser@<DVC_HOST>:2222/srv/dvc-storage
```

`dvc-write` queda almacenado en:

```text
.dvc/config.local
```

Este archivo no debe versionarse en Git.

---

## 7. Regla importante sobre el remoto por defecto

El remoto predeterminado del proyecto es:

```text
dvc-public
```

por lo tanto:

```bash
dvc pull
```

utiliza HTTP público.

Sin embargo, el servidor HTTP es deliberadamente de solo lectura.

Por esta razón, los escritores deben subir siempre especificando explícitamente:

```bash
dvc push -r dvc-write
```

No utilizar:

```bash
dvc push
```

porque intentaría escribir contra `dvc-public` y el servidor HTTP rechazará la operación.

Esta separación es intencional.

---

## 8. Primera carga en un servidor vacío

El servidor creado en el Manual 1 comienza sin objetos DVC.

Para realizar la primera carga se necesita una máquina escritora que posea físicamente los datos que se desean almacenar.

Los archivos deben estar bajo seguimiento de DVC antes de ejecutar el `push`.

### Caso A — Los archivos ya están bajo seguimiento de DVC

Verificar:

```bash
dvc status
```

Subir los objetos disponibles en el cache local:

```bash
dvc push -r dvc-write
```

### Caso B — Incorporar un archivo nuevo

Ejemplo:

```bash
dvc add data/raw/foorilla/jobs_<CORTE>.csv
```

Subir el objeto:

```bash
dvc push -r dvc-write
```

Después versionar la metadata en Git:

```bash
git add .
git commit -m "Add dataset snapshot"
git push
```

La regla es:

```text
archivo grande
    ↓
dvc add
    ↓
dvc push -r dvc-write
    ↓
metadata DVC
    ↓
git commit / git push
```

---

## 9. Verificar la carga en el servidor

En el servidor:

```bash
find /srv/dvc-storage -type f | head
```

Deben aparecer objetos con rutas internas administradas por DVC, por ejemplo:

```text
/srv/dvc-storage/files/md5/ab/...
/srv/dvc-storage/files/md5/c3/...
```

No es necesario ni esperado encontrar los archivos con sus nombres originales.

---

## 10. Verificar lectura HTTP

Seleccionar en el servidor un objeto existente:

```bash
FILE=$(find /srv/dvc-storage -type f | head -n 1)
```

Obtener su ruta relativa:

```bash
RELATIVE_PATH=$(realpath \
  --relative-to=/srv/dvc-storage \
  "$FILE")

echo "$RELATIVE_PATH"
```

Desde cualquier máquina con acceso HTTP:

```bash
curl -I \
  "http://<DVC_HOST>:8080/dvc/${RELATIVE_PATH}"
```

La respuesta debe incluir normalmente:

```text
HTTP/1.1 200 OK
```

---

## 11. Validar `dvc pull` sin credenciales

La validación más importante debe realizarse desde un clone limpio para evitar que el cache local oculte problemas.

En otra carpeta o máquina:

```bash
git clone <URL_REPOSITORIO> salarypredict-clean
cd salarypredict-clean/ml
```

Preparar DVC:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.lock.txt
```

Comprobar el remoto:

```bash
dvc remote list
```

Debe aparecer `dvc-public`.

Ejecutar:

```bash
dvc pull
```

La descarga debe completarse sin solicitar:

```text
usuario
contraseña
OAuth
AWS credentials
SSH key
```

Verificar:

```bash
ls data/raw/foorilla/
```

Los snapshots versionados deben quedar materializados localmente.

---

## 12. Validar escritura desde un cliente autorizado

En una máquina que tenga configurado `dvc-write`:

```bash
dvc push -r dvc-write
```

Una ejecución sin cambios puede responder que todo se encuentra actualizado.

Si la clave privada utiliza passphrase, dicha passphrase debe gestionarse localmente y nunca almacenarse en Git.

---

## 13. Flujo cotidiano de un consumidor

Después de clonar el repositorio:

```bash
cd Microproyecto/ml
dvc pull
```

Para actualizar código y datos:

```bash
git pull
dvc pull
```

No necesita credenciales del servidor DVC.

---

## 14. Flujo cotidiano de un escritor

Primero actualizar:

```bash
git pull
dvc pull
```

Para un nuevo snapshot:

```bash
dvc add data/raw/foorilla/jobs_<CORTE>.csv
```

Subir el objeto físico:

```bash
dvc push -r dvc-write
```

Versionar metadata y código:

```bash
git add .
git commit -m "Add dataset snapshot"
git push
```

Los demás integrantes recuperan posteriormente:

```bash
git pull
dvc pull
```

---

## 15. Qué se versiona y qué no

### Se versiona en Git

```text
.dvc/config
*.dvc
dvc.yaml
dvc.lock
params.yaml
```

según corresponda al proyecto.

El remoto público forma parte de `.dvc/config`.

### No se versiona

```text
.dvc/config.local
claves privadas SSH
authorized_keys
datasets físicos
cache local de DVC
```

El remoto privado `dvc-write` debe permanecer en `.dvc/config.local`.

---

## 16. Incorporar un nuevo escritor

Primero, el servidor debe contener su clave pública en:

```text
~/dvc-server/ssh/authorized_keys
```

Una vez autorizada y reiniciado `dvc-ssh`, el nuevo integrante configura:

```bash
dvc remote add --local dvc-write \
  ssh://dvcuser@<DVC_HOST>:2222/srv/dvc-storage
```

y su propia clave privada:

```bash
dvc remote modify --local dvc-write \
  keyfile ~/.ssh/<RUTA_DE_SU_CLAVE_PRIVADA>
```

Cada escritor conserva su propia clave privada.

---

## 17. Retirar acceso de escritura

Eliminar del servidor la clave pública correspondiente:

```bash
nano ~/dvc-server/ssh/authorized_keys
```

Reiniciar:

```bash
cd ~/dvc-server
docker compose restart dvc-ssh
```

El usuario pierde capacidad de `dvc push`, pero puede continuar ejecutando:

```bash
dvc pull
```

porque la lectura HTTP permanece pública.

---

## 18. Troubleshooting básico

### `Permission denied (publickey)`

Verificar primero:

```bash
ssh \
  -i ~/.ssh/salarypredict/dvc_server_key \
  -p 2222 \
  dvcuser@<DVC_HOST>
```

Si SSH directo no funciona, DVC tampoco podrá hacer `push`.

---

### DVC no reconoce el backend SSH

Instalar:

```bash
pip install "dvc[ssh]"
```

---

### `dvc push` falla contra HTTP

Si se ejecuta:

```bash
dvc push
```

DVC utilizará `dvc-public`, que es de solo lectura.

Usar:

```bash
dvc push -r dvc-write
```

---

### `dvc pull` devuelve HTTP 404

Comprobar que el servidor fue previamente poblado:

```bash
find /srv/dvc-storage -type f | head
```

y que el remoto tenga exactamente esta base:

```text
http://<DVC_HOST>:8080/dvc/
```

---

### `dvc pull` funciona en el repositorio original pero no en un clone limpio

El repositorio original puede estar resolviendo los objetos desde su cache local.

La prueba válida es:

```text
clone limpio
→ instalar DVC
→ dvc pull
```

---

## 19. Criterios de aceptación

La configuración completa se considera válida cuando:

```text
[ ] dvc-public está versionado como remoto por defecto
[ ] dvc pull funciona desde un clone limpio
[ ] dvc pull no solicita credenciales
[ ] dvc-write existe solo en .dvc/config.local
[ ] SSH directo a dvcuser@<DVC_HOST>:2222 funciona
[ ] dvc push -r dvc-write sube objetos correctamente
[ ] HTTP rechaza operaciones de escritura
[ ] otro cliente recupera por HTTP los objetos subidos por SSH
```

---

## Estado final

```text
Repositorio Git
│
├── .dvc/config
│   └── dvc-public
│       └── http://<DVC_HOST>:8080/dvc/
│           └── default
│
└── .dvc/config.local
    └── dvc-write
        └── ssh://dvcuser@<DVC_HOST>:2222/srv/dvc-storage
            └── solo clientes autorizados
```

Uso final:

```bash
# Todos los usuarios
dvc pull

# Solo escritores autorizados
dvc push -r dvc-write
```

De esta forma la lectura permanece anónima y simple, mientras que la escritura conserva autenticación mediante claves SSH y no expone credenciales dentro del repositorio.

Para la guía general de despliegue y bootstrap del proyecto, consulte [`../../DEPLOYMENT.md`](../../DEPLOYMENT.md).
