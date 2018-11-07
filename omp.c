#define PNG_NO_SETJMP

#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX_ITER 10000

void write_png(const char* filename, const int width, const int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
	
	for (int y = 0; y < height; ++y) {			
		memset(row, 0, row_size);
		for (int x = 0; x < width; ++x) {
			int p = buffer[(height - 1 - y) * width + x];
			png_bytep color = row + x * 3;
			if (p != MAX_ITER) {
				if (p & 16) {
					color[0] = 240;
					color[1] = color[2] = p % 16 * 16;	
				} else {
					color[0] = p % 16 * 16;
				}
			}
		}
		png_write_row(png_ptr, row);
	}
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int MondebrotSet(double x0, double y0)
{
	int repeats = 1;
	double x = x0;
	double y = y0;
	double r2 = x * x, i2 = y * y;
	if(r2 + i2 < 0.0625)
		return MAX_ITER;
	for( ; repeats < MAX_ITER ; ++repeats)
	{
		r2 = x * x;
		i2 = y * y;
		if(r2 + i2 > 4.0)
			return repeats;
		y = 2.0 * x * y + y0;
		x = r2 - i2 + x0;
	}
	return repeats;
}

int main(int argc, char** argv) {
	
    /* argument parsing */
    assert(argc == 9);
    int num_threads = strtol(argv[1], 0, 10);
    double left = strtod(argv[2], 0);
    double right = strtod(argv[3], 0);
    double lower = strtod(argv[4], 0);
    double upper = strtod(argv[5], 0);
    int width = strtol(argv[6], 0, 10);
    int height = strtol(argv[7], 0, 10);
    const char* filename = argv[8];

	double x[width], y[height];
	int done[height];
	
	omp_set_nested(1);
	
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic) nowait
		for(int i=0 ; i<width ; ++i)
			x[i] = i * ((right - left) / width) + left;
	}
	
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic) nowait
		for(int j=0 ; j<height ; ++j)
		{
			y[j] = j * ((upper - lower) / height) + lower;
			done[j] = 0;
		}
	}
	
	/* allocate memory for image */
	int* image = (int*)malloc(width * height * sizeof(int));
	assert(image);
	
	
	/* mandelbrot set */
	#pragma omp parallel
	{	
		#pragma omp for schedule(dynamic) collapse(2) nowait
		for (int j = 0; j < height; ++j) {
			for (int i = 0; i < width; ++i) {
				image[j*width+i] = MondebrotSet(x[i], y[j]);
			}
		}
	}
		
	/* draw and cleanup */
	write_png(filename, width, height, image);
	free(image);
}
