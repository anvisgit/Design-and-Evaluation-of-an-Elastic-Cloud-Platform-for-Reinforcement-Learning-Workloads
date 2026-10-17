SHELL := /bin/bash
STAGE ?= dev
ACCOUNT ?= $(shell aws sts get-caller-identity --query Account --output text)
REGION ?= $(shell aws configure get region)
GIT_SHA ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
REPO ?= $(ACCOUNT).dkr.ecr.$(REGION).amazonaws.com/rl-platform-worker

.PHONY: install layer test test-all synth push-image deploy destroy lint

install:
	pip install -r requirements-dev.txt

layer:
	rm -rf build/layer && mkdir -p build/layer/python
	pip install -r lambdas/requirements.txt --target build/layer/python \
	  --platform manylinux2014_aarch64 --only-binary=:all: --python-version 3.11 --implementation cp

test:
	python -m pytest tests/unit -q

test-all:
	python -m pytest tests/unit tests/integration -q

synth: layer
	cdk synth --context stage=$(STAGE)

push-image:
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ACCOUNT).dkr.ecr.$(REGION).amazonaws.com
	docker build -f worker/Dockerfile --build-arg GIT_SHA=$(GIT_SHA) -t $(REPO):$(GIT_SHA) -t $(REPO):latest .
	docker push $(REPO):$(GIT_SHA)
	docker push $(REPO):latest

deploy:
	./scripts/deploy.sh $(STAGE)

destroy:
	-aws s3 rm s3://rl-platform-artifacts-$(ACCOUNT)-$(REGION) --recursive
	cdk destroy --all --force --context stage=$(STAGE)
