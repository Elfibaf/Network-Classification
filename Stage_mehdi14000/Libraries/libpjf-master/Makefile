CFLAGS =
LDFLAGS = -lm

ME=libpjf
C_OBJECTS=lib.o regex.o thash.o tlist.o xstr.o mmatic.o \
	unitype.o sfork.o json.o utf8.o

TARGETS=libpjf.so libpjf.a

include rules.mk

libpjf.so: $(C_OBJECTS)
	$(CC) $(C_OBJECTS) -shared -o libpjf.so $(LDFLAGS)

libpjf.a: $(C_OBJECTS)
	$(AR) rc libpjf.a $(C_OBJECTS)

install: install-std

utilities:
	$(MAKE) -C utils

distclean: clean
	$(MAKE) -C distclean
