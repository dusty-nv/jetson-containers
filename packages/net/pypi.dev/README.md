# devpi.dev

#### This is an example of a private PyPI server using devpi.dev folder structure with docker-compose.

#### Run your own devpi server with a simple Docker Compose. 
#### This setup allows you to run a private PyPI server for package management.

# Usage
```bash
docker compose --env-file .env -f ./services/docker-compose.yml up -d
```

#### Open your browser and go to `http://localhost:3141` to access the devpi server.
