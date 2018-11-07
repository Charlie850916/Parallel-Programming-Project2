#define PNG_NO_SETJMP

#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "mpi.h"

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

    MPI_Init(&argc, &argv);
	
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
	
    int rank, size;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	double x[width], y[height];
	
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
			y[j] = j * ((upper - lower) / height) + lower;
	}

	if(size==1)
	{
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
	else
	{
		if(rank==0) // master
		{
			/* allocate memory for image */
			int* image = (int*)malloc(width * height * sizeof(int));
			assert(image);
		
			MPI_Request req, reqArr[height];
			int check[height];
			
			#pragma omp parallel
			{
				#pragma omp for schedule(dynamic) nowait
				for(int i=0 ; i<height ; ++i)
					check[i] = 0;
			}
		
			int count = size-1, j = size-1, info[2]; // [row rank];
			if(height < count)
				count = height;
			
			omp_lock_t lock;
			omp_init_lock(&lock);
			
			omp_set_nested(1);
			
			#pragma omp parallel 
			{
				#pragma omp master
				{
					do
					{
						MPI_Recv(info, 2, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						int k = height-1-info[0];
						MPI_Irecv(image+k*width, width, MPI_INT, info[1], 1, MPI_COMM_WORLD, &reqArr[k]);
						check[k] = 1;
						count--;
						omp_set_lock(&lock);
						MPI_Isend(&j, 1, MPI_INT, info[1], 0, MPI_COMM_WORLD, &req);
						if(j < height)
						{
							count++;
							j++;
						}
						omp_unset_lock(&lock);
					}
					while(count);
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
						int k = height-1-y, val = 0;
						if(check[k])
							MPI_Wait(&reqArr[k], MPI_STATUS_IGNORE);
						for (int x = 0; x < width; ++x) {
							int p = image[k * width + x];
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
				
				#pragma omp single
				{
					int cur;
					omp_set_lock(&lock);
					cur = j++;
					omp_unset_lock(&lock);
					while(cur < height)
					{
						int k = height-1-cur;
						#pragma omp parallel
						{
							#pragma omp for schedule(dynamic) nowait
							for (int i = 0; i < width; ++i) {
								image[k*width+i] = MondebrotSet(x[i], y[k]);
							}
						}
						omp_set_lock(&lock);
						cur = j++;
						omp_unset_lock(&lock);
					}
				}
			}
			
			omp_destroy_lock(&lock);
			free(image);
		}
		else // slave
		{
			MPI_Request req;
			
			int row = rank-1, info[2];
			int* tmp = (int*)malloc(width * height * sizeof(int));
			info[1] = rank;

			while(row < height)
			{
				int k = height-1-row;
				#pragma omp parallel
				{
					#pragma omp for schedule(dynamic) nowait
					for (int i = 0; i < width; ++i) {
						tmp[k*width+i] = MondebrotSet(x[i], y[k]);
					}
				}
				info[0] = row;
				MPI_Isend(info, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
				MPI_Isend(tmp+k*width, width, MPI_INT, 0, 1, MPI_COMM_WORLD, &req);
				MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
	}
    MPI_Finalize();
}
