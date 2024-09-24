SHELL := /bin/zsh

.PHONY sync:
sync:
	@echo "Syncing files from the vault..."
	@cp -r ../../../Documents/Obsidian/Creation/Blog/drafts/*.md ./pages/_drafts/
	@cp -r ../../../Documents/Obsidian/Creation/Blog/stories/*.md ./pages/_stories/
	@cp -r ../../../Documents/Obsidian/Creation/Blog/articles/*.md ./pages/_articles/
	@cp -r ../../../Documents/Obsidian/Creation/Blog/projects/*.md ./pages/_projects/
	@echo "Syncing images from the vault..."
	@cp -r ../../../Documents/Obsidian/Creation/Blog/(drafts|stories|articles|projects)/*.(png|webp|svg) ./assets/images/
	@echo "Done."
