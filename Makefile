.PHONY: up down logs test test-docker avaliar install run reset

# ---------- Docker ----------
up:            ## sobe API + PostgreSQL
	docker compose up --build -d
	@echo "Swagger: http://localhost:8000/docs"

down:          ## para os containers
	docker compose down

logs:
	docker compose logs -f api

test-docker:   ## roda os testes dentro do container
	docker compose run --rm --no-deps --entrypoint pytest api

reset:         ## apaga o banco e os dados baixados
	docker compose down -v

# ---------- Local ----------
install:
	./scripts/instalar.sh

run:
	./scripts/executar.sh

test:
	python -m pytest

avaliar:       ## avaliação offline (Precision@10, Recall@10, nDCG@10)
	python -m scripts.avaliar
