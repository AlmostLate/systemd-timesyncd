all: build
.PHONY: all

build:
	go build -o bin/ ./cmd/psb-backend
.PHONY: build

clean:
	rm -rf bin/
.PHONY: clean

fmt:
	go fmt ./...
.PHONY: fmt

vet:
	go vet ./...
	golangci-lint run --default=all --disable="depguard"
.PHONY: vet

generate-server:
	mkdir -p generated
	swagger generate server \
    -f ./spec/v1/openapi.yaml \
    --target ./generated \
    --principal=interface{} \
    --exclude-main
.PHONY: generate-server
