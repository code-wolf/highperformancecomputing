CC=g++
TARGET=gauss
CFLAGS=
LIBS=-framework OpenCL
OBJ=main.o tga.o image_utils.o

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(TARGET): $(OBJ)
	$(CC) $(LIBS) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o *~ $(TARGET)