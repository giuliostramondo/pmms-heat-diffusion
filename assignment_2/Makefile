all: $(filter-out Makefile README.md,$(wildcard *))

.PHONY: $(filter-out Makefile README.md,$(wildcard *)) clean $(addprefix clean,$(filter-out Makefile README.md,$(wildcard *)))

$(filter-out Makefile README.md,$(wildcard *)):
	$(MAKE) -C $@

clean: $(addprefix clean_,$(filter-out Makefile README.md,$(wildcard *)))
	@echo $+
	

$(addprefix clean_,$(filter-out Makefile README.md,$(wildcard *))):
	@echo clean $@
	$(MAKE) -C $(subst clean_,,$@) clean
