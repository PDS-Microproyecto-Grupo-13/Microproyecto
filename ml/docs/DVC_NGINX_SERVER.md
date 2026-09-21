# Servidor DVC Autogestionado con Docker, Nginx y SSH

Guía operativa para desplegar un servidor DVC temporal y reproducible con dos canales de acceso sobre el mismo almacenamiento:

- **Lectura pública por HTTP** mediante Nginx, sin credenciales.
- **Escritura autenticada por SSH** mediante claves públicas.

Este documento cubre únicamente la preparación y validación inicial del servidor, hasta dejarlo listo para recibir la primera carga desde el remoto DVC actual.

---

## 1. Arquitectura

```text
                         Internet
                            │
             ┌──────────────┴──────────────┐
             │                             │
        SSH :2222                    HTTP :8080
     autenticado por key             sin credenciales
             │                             │
             v                             v
        dvc-ssh                       dvc-http
          RW                              RO
             │                             │
             └────────────┬────────────────┘
                          │
                          v
                  /srv/dvc-storage
                    host / EBS
```

Ambos contenedores comparten el mismo almacenamiento físico:

```text
/srv/dvc-storage
```

con permisos distintos:

```text
dvc-ssh   -> lectura/escritura
dvc-http  -> solo lectura
```

Los datos permanecen en el host aunque los contenedores sean reconstruidos.

---

## 2. Requisitos

Servidor Linux, por ejemplo Ubuntu 22.04 o 24.04, con:

- Docker Engine
- Docker Compose v2
- acceso administrativo al host

Verificar:

```bash
docker --version
docker compose version
```

Puertos utilizados:

| Puerto | Uso |
|---|---|
| `22` | SSH administrativo del servidor |
| `2222` | SSH para `dvc push` |
| `8080` | HTTP público para `dvc pull` |

---

## 3. Crear la estructura del servidor

En el host:

```bash
mkdir -p ~/dvc-server/nginx
mkdir -p ~/dvc-server/ssh

cd ~/dvc-server
```

Estructura esperada:

```text
dvc-server/
├── compose.yaml
├── nginx/
│   └── default.conf
└── ssh/
    ├── Dockerfile
    ├── entrypoint.sh
    ├── sshd_config
    └── authorized_keys
```

El almacenamiento DVC no se guarda dentro de esta carpeta.

---

## 4. Crear el almacenamiento persistente

Se utilizará un UID/GID dedicado:

```text
10001
```

Crear el directorio:

```bash
sudo mkdir -p /srv/dvc-storage
sudo chown -R 10001:10001 /srv/dvc-storage
sudo chmod 755 /srv/dvc-storage
```

No copiar manualmente datasets dentro de este directorio. DVC administrará su propia estructura interna.

---

## 5. Configurar Nginx

Crear:

```bash
nano nginx/default.conf
```

Contenido:

```nginx
server {
    listen 80 default_server;
    server_name _;

    location = /healthz {
        default_type text/plain;
        return 200 "ok\n";
    }

    location /dvc/ {
        alias /srv/dvc-storage/;

        autoindex off;

        # Lectura pública solamente.
        # GET también habilita HEAD.
        limit_except GET {
            deny all;
        }
    }
}
```

El endpoint público será:

```text
http://<DVC_HOST>:8080/dvc/
```

---

## 6. Crear la imagen SSH

Crear:

```bash
nano ssh/Dockerfile
```

Contenido:

```dockerfile
FROM ubuntu:24.04

ARG DVC_UID=10001
ARG DVC_GID=10001

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        --no-install-recommends \
        openssh-server \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${DVC_GID} dvc \
    && useradd \
        --uid ${DVC_UID} \
        --gid ${DVC_GID} \
        --create-home \
        --shell /bin/bash \
        dvcuser \
    # Enable the Unix account for SSH public-key authentication.
    # Password login remains disabled by sshd_config.
    && passwd -d dvcuser \
    && mkdir -p /run/sshd \
    && mkdir -p /home/dvcuser/.ssh \
    && mkdir -p /config \
    && chown -R dvcuser:dvc /home/dvcuser

COPY sshd_config /etc/ssh/sshd_config
COPY entrypoint.sh /usr/local/bin/dvc-ssh-entrypoint

RUN chmod 755 /usr/local/bin/dvc-ssh-entrypoint

EXPOSE 22

ENTRYPOINT ["/usr/local/bin/dvc-ssh-entrypoint"]
```

---

## 7. Configurar SSH

Crear:

```bash
nano ssh/sshd_config
```

Contenido:

```text
Port 22
ListenAddress 0.0.0.0

PermitRootLogin no

PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes

AuthorizedKeysFile .ssh/authorized_keys

UsePAM no

X11Forwarding no
AllowTcpForwarding no
AllowAgentForwarding no
PermitTunnel no

PermitUserEnvironment no
PrintMotd no

StrictModes yes

Subsystem sftp internal-sftp

AllowUsers dvcuser
```

Esta configuración deja habilitado únicamente el acceso por clave pública para `dvcuser`.

---

## 8. Crear el entrypoint SSH

Crear:

```bash
nano ssh/entrypoint.sh
```

Contenido:

```bash
#!/usr/bin/env bash

set -euo pipefail

AUTHORIZED_KEYS_SOURCE="/config/authorized_keys"
AUTHORIZED_KEYS_TARGET="/home/dvcuser/.ssh/authorized_keys"

if [ ! -s "${AUTHORIZED_KEYS_SOURCE}" ]; then
    echo "ERROR: /config/authorized_keys does not exist or is empty." >&2
    exit 1
fi

install \
    -d \
    -m 700 \
    -o dvcuser \
    -g dvc \
    /home/dvcuser/.ssh

install \
    -m 600 \
    -o dvcuser \
    -g dvc \
    "${AUTHORIZED_KEYS_SOURCE}" \
    "${AUTHORIZED_KEYS_TARGET}"

ssh-keygen -A

# Los objetos nuevos deben ser legibles por Nginx.
umask 022

exec /usr/sbin/sshd -D -e
```

El Dockerfile asignará permisos de ejecución al script.

---

## 9. Agregar claves públicas autorizadas

Crear:

```bash
touch ssh/authorized_keys
```

Cada integrante autorizado genera su propia clave:

```bash
ssh-keygen -t ed25519
```

Archivos típicos:

```text
~/.ssh/id_ed25519
~/.ssh/id_ed25519.pub
```

Solo debe compartirse la clave pública:

```text
id_ed25519.pub
```

Agregar una clave por línea:

```bash
nano ssh/authorized_keys
```

Ejemplo:

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... usuario1
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... usuario2
```

No versionar `authorized_keys` en Git.

---

## 10. Crear Docker Compose

Crear:

```bash
nano compose.yaml
```

Contenido:

```yaml
services:

  dvc-http:
    image: nginx:alpine
    container_name: dvc-http
    restart: unless-stopped

    ports:
      - "8080:80"

    volumes:
      - /srv/dvc-storage:/srv/dvc-storage:ro
      - ./nginx/default.conf:/etc/nginx/conf.d/default.conf:ro


  dvc-ssh:
    build:
      context: ./ssh
      args:
        DVC_UID: 10001
        DVC_GID: 10001

    container_name: dvc-ssh
    restart: unless-stopped

    ports:
      - "2222:22"

    volumes:
      - /srv/dvc-storage:/srv/dvc-storage
      - ./ssh/authorized_keys:/config/authorized_keys:ro
```

La diferencia importante es:

```text
dvc-http -> storage montado read-only
dvc-ssh  -> storage montado read-write
```

---

## 11. Validar la configuración

Antes de iniciar:

```bash
docker compose config
```

La configuración debe resolverse sin errores.

---

## 12. Construir e iniciar los servicios

```bash
docker compose up -d --build
```

Verificar:

```bash
docker compose ps
```

Estado esperado:

```text
dvc-http   Up
dvc-ssh    Up
```

---

## 13. Revisar logs

Nginx:

```bash
docker compose logs dvc-http
```

SSH:

```bash
docker compose logs dvc-ssh
```

Seguimiento conjunto:

```bash
docker compose logs -f
```

---

## 14. Probar el servicio HTTP

Desde el propio servidor:

```bash
curl http://localhost:8080/healthz
```

Respuesta esperada:

```text
ok
```

Desde otra máquina:

```bash
curl http://<DVC_HOST>:8080/healthz
```

También debe responder:

```text
ok
```

---

## 15. Probar el acceso SSH

Desde una máquina cuya clave pública esté autorizada:

```bash
ssh -p 2222 dvcuser@<DVC_HOST>
```

Comprobar:

```bash
whoami
```

Resultado esperado:

```text
dvcuser
```

Salir:

```bash
exit
```

---

## 16. Verificar escritura en el storage

Desde un cliente autorizado:

```bash
ssh -p 2222 dvcuser@<DVC_HOST> \
  'touch /srv/dvc-storage/.write-test'
```

Verificar:

```bash
ssh -p 2222 dvcuser@<DVC_HOST> \
  'ls -l /srv/dvc-storage/.write-test'
```

Eliminar:

```bash
ssh -p 2222 dvcuser@<DVC_HOST> \
  'rm /srv/dvc-storage/.write-test'
```

Esto confirma que el canal SSH tiene permisos de escritura.

---

## 17. Verificar que HTTP sea solo lectura

Desde cualquier máquina:

```bash
curl -i \
  -X PUT \
  --data "test" \
  http://<DVC_HOST>:8080/dvc/test.txt
```

La respuesta debe ser un código HTTP no exitoso, por ejemplo:

```text
403 Forbidden
```

Verificar que el archivo no exista:

```bash
curl http://<DVC_HOST>:8080/dvc/test.txt
```

---

## 18. Configurar firewall o Security Group

Configuración recomendada:

| Puerto | Exposición recomendada | Propósito |
|---|---|---|
| `22` | IP administrativa | administración del host |
| `2222` | IPs del equipo | `dvc push` autenticado |
| `8080` | público | `dvc pull` sin credenciales |

Ejemplo conceptual:

```text
TCP 22
source = TU_IP/32

TCP 2222
source = IP_EQUIPO_1/32
source = IP_EQUIPO_2/32

TCP 8080
source = 0.0.0.0/0
```

No es necesario abrir otros puertos para este servicio.

## Estado esperado al finalizar

El servidor debe quedar con:

- `dvc-http` operativo en `:8080`
- `dvc-ssh` operativo en `:2222`
- `/srv/dvc-storage` persistente
- acceso SSH por clave pública
- escritura habilitada solo por SSH
- HTTP público en modo solo lectura
