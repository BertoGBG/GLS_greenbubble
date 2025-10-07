# Convenience targets
.PHONY: lock
lock:
	conda-lock -f envs/environment.base.yml \	  -p linux-64 -p osx-64 -p osx-arm64 -p win-64 \	  -k explicit -o envs/locks/

.PHONY: install-linux
install-linux:
	conda-lock install -n greenbubble_gls-linux envs/locks/conda-linux-64.lock

.PHONY: install-osx-arm
install-osx-arm:
	conda-lock install -n greenbubble_gls-osxarm envs/locks/conda-osx-arm64.lock

.PHONY: install-osx-64
install-osx-64:
	conda-lock install -n greenbubble_gls-osx envs/locks/conda-osx-64.lock

.PHONY: install-win
install-win:
	@echo "Run the following in an Anaconda Prompt on Windows:"
	@echo conda-lock install -n greenbubble_gls-win envs/locks/conda-win-64.lock
