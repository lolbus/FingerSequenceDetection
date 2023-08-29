import multiprocessing

max_threads = multiprocessing.cpu_count()
print(f"The maximum number of threads is: {max_threads}")