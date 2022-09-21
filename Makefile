CC = gcc
LIBS = -lm
CFLAGS = -fstack-protector-strong -march=native
OPTIMIZATIONFLAGS = -Ofast
PROFILEFLAGS = -pg
DEBUGFLAGS = -g

default:
	$(CC) run.c -o build/run $(CFLAGS) $(LIBS) $(OPTIMIZATIONFLAGS)

debug:
	$(CC) run.c -o build/run $(CFLAGS) $(LIBS) $(DEBUGFLAGS)

profile:
	$(CC) run.c -o build/run $(CFLAGS) $(LIBS) $(PROFILEFLAGS)
	build/run
	gprof build/run gmon.out > perform.txt

clean:
	rm build/* gmon.out perform.txt core*
