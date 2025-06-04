# === Config ===
USER = axelle
HOST = idun-login2.hpc.ntnu.no
REMOTE_PATH = FG-FNC
VENV_PATH = ~/venvs/fgfnc-venv

# === Targets ===
ssh:
	ssh $(USER)@$(HOST) -t "cd $(REMOTE_PATH) && exec \$$SHELL"

push:
	bash sync-to-idun.sh

pull:
	@test -n "$(path)" || (echo "path is not set. Usage: make pull path=path/to/file.txt" && exit 1)
	rsync -av $(USER)@$(HOST):/cluster/home/$(USER)/$(REMOTE_PATH)/$(path) ./$(path)
