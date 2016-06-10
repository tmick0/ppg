ppg/ppgfilter.so: c_module/ppgfilter.so
	cp $< $@

c_module/ppgfilter.so: c_module
	$(MAKE) -C c_module

.DUMMY: clean
clean:
	$(MAKE) -C c_module clean
