
.DUMMY: module clean

module:
	$(MAKE) -C ppgfilter

clean:
	$(MAKE) -C ppgfilter clean
