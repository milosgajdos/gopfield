EXDIR=example
BINARY=gopfield
BUILD=go build
CLEAN=go clean
INSTALL=go install
BUILDPATH=./_build
PACKAGES=$(shell go list ./... | grep -vE /vendor/)

examples: builddir
	$(BUILD) -v -o $(BUILDPATH)/gopfield $(EXDIR)/main.go

all: examples

install:
	$(INSTALL) ./...

clean:
	rm -rf $(BUILDPATH)
	rm -rf $(GOPATH)/bin/$(BINARY)

builddir:
	mkdir -p $(BUILDPATH)

check:
	for pkg in ${PACKAGES}; do \
		go vet $$pkg || exit ; \
		golint $$pkg || exit ; \
	done

test:
	for pkg in ${PACKAGES}; do \
		go test -coverprofile="../../../$$pkg/coverage.txt" -covermode=atomic $$pkg || exit; \
	done

.PHONY: clean examples
