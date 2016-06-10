ppg/ppgfilter.so: c_module/ppgfilter.so
	cp $< $@

c_module/ppgfilter.so: c_module
	$(MAKE) -C c_module

.PHONY: clean docs

clean:
	$(MAKE) -C c_module clean

docs:
	mkdir -p docs
	PYTHONPATH=. pdoc --html --html-no-source --html-dir docs --overwrite --external-links ppg
