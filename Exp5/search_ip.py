import argparse
import random
import threading
import time
import gc
import numpy as np
import csv
import datetime
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
# connections.add_connection(default={"host": "10.255.156.187", "port": "19530"})
# connections.add_connection(default={"host": "10.255.53.192", "port": "19530"})
connections.add_connection(default={"host": "10.255.199.200", "port": "19530"})
connections.connect("default")

NQ = 1
TopK = 50
ef_search = 64

rst = {}

exp_runs = 1
warm_up_runs = 50
runs = 200

time_tick_interval = [100, 200, 400, 600, 800, 1000]
# time_tick_interval = [100, 200, 400, 600]
# time_tick_interval = [800, 1000]

graceful_times_dict = {}
for t in time_tick_interval:
    graceful_times_dict[t] = []
    for g in range(0, 3*t+1, int(t*0.3)):
        graceful_times_dict[t].append(g)
print("settings, TTI->[GT]", graceful_times_dict)

def time_costing(func):
    def core(*args):
        start = time.time()
        print(func.__name__, "start time: ", start)
        res = func(*args)
        end = time.time()
        print(func.__name__, "end time: ", end)
        print(func.__name__, "time cost: ", end-start)
        return res
    return core


# def create_collection(collection_name, field_name, dim, partition=None, auto_id=True):
#     pk = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=auto_id)
#     field = FieldSchema(name=field_name, dtype=DataType.FLOAT_VECTOR, dim=dim)
#     schema = CollectionSchema(fields=[pk, field], description="example collection")
#
#     collection = Collection(name=collection_name, schema=schema)
#     return collection


# @time_costing
# def create_index(collection, field_name):
    # default_index = {"index_type": "IVF_FLAT", "params": {"nlist": 1024}, "metric_type": "L2"}
    # collection.create_index(field_name, default_index)
    # print("Successfully build index")
    # print(pymilvus.utility.index_building_progress(collection.name))

    # collection.drop_index()
    # print("Successfully drop index")


# @time_costing
def search(collection, query_entities, field_name, topK, guarantee_timestamp=0):
    search_params = {"metric_type": "IP", "params": {"ef": ef_search}}
    res = collection.search(query_entities, field_name, search_params, limit=topK,
                            guarantee_timestamp=guarantee_timestamp)


# @time_costing
# def insert(collection, entities, partition):
#     mr = collection.insert([entities], partition_name=partition)
#     print(mr)


# def gen_data_and_insert(collection, nb, batch, dim, partition):
#     for i in range(int(nb/batch)):
#         entities = generate_entities(dim, nb)
#         insert(collection, entities, partition)
#         gc.collect()
#
# def insert_batch(collection, nb, dim, batch, thread_num=1):
#     for i in range(int(nb/batch)):
#         entities = generate_entities(dim, batch)
#         insert(collection, entities, None)
#         gc.collect()
#
#
# def insert_parallel(collection, partition, dim, batch, speed):
#     while True:
#         global stop_insert
#         if stop_insert:
#             return
#         entities = generate_entities(dim, batch)
#         insert_start = time.time()
#         insert(collection, entities, partition)
#         insert_end = time.time()
#         if speed < (insert_end-insert_start):
#             raise Exception("Speed if too small")
#         time.sleep(speed-(insert_end-insert_start))
#         gc.collect()


def generate_entities(dim, nb) -> list:
    vectors = [[random.random() for _ in range(dim)] for _ in range(nb)]
    return vectors


# def insert_data_from_file(coll, nb, dim,  vectors_per_file, batch_size):
#     # logger.info("Load npy file: %s end" % file_name)
#     for j in range(nb // vectors_per_file):
#         s = "%05d" % j
#         fname = "binary_" + str(dim) + "d_" + s + ".npy"
#         data = np.load(fname)
#         vectors = data.tolist()
#         insert(coll, vectors)


def graceful_time_search(coll, field_name, graceful_time, tti):
    # print("warming up", graceful_time)
    # print("graceful_time", graceful_time, flush=True)

    tti /= 1000.0

    for i in range(warm_up_runs):
        query_entities = generate_entities(dim, NQ)
        time.sleep(tti+random.uniform(0, tti))
        search(coll, query_entities, field_name, TopK)

    # print("time tick interval = ", time_tick_interval, "graceful time = ", graceful_time, "start time = ", time.time())

    total_time = 0
    for i in range(runs):
        query_entities = generate_entities(dim, NQ)
        time.sleep(tti+random.uniform(0, tti))
        t0 = time.time()
        search(coll, query_entities, field_name, TopK)
        total_time += (time.time() - t0)
    rst[graceful_time].append(total_time/runs)
    print(rst[graceful_time][-1])
    # print("time tick interval = ", time_tick_interval, "graceful time = ", graceful_time, "end time = ", time.time(), flush=True)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="your script description")  # description参数可以用于插入描述脚本用途的信息，可以为空
    # parser.add_argument('--speed', '-s', nargs='?', type=float, help='insert speed, :s', default=0.1)
    # parser.add_argument('--batch', '-b', nargs='?', type=int, help='insert batch num', default=10)
    # args = parser.parse_args()

    # batch = args.batch
    # speed = args.speed

    collection_name = "deep_100m_96_ip"
    field_name = "float_vector"
    dim = 96

    if not utility.has_collection(collection_name):
        raise Exception(collection_name, "does not exist")

    coll = Collection(collection_name)

    # partition_name = "cat"
    # coll.create_partition(partition_name)

    # insert_batch(coll, 1000000, dim, 50000)
    # create_index(coll, field_name)

    coll.load()
    print("partitions: ", coll.partitions)

    # stop_insert = False
    # t1 = threading.Thread(target=insert_parallel, args=(coll, partition_name, dim, batch, speed))
    # t1.start()

    # input("Press Enter to confirm insert has started...")

    print("Start varying graceful_time and search...")

    for tti in graceful_times_dict:
        coll.set_timetick_interval(tti)
        time.sleep(10)
        rst = {}

        for gt in graceful_times_dict[tti]:
            rst[gt] = []
            for r in range(exp_runs):
                print("[{0}] TTI: {1} GT: {2} Run: {3}".format(datetime.datetime.now().time(), tti, gt, r), flush=True)

                if r == 0:
                    print("set gt: ", gt)
                    coll.set_graceful_time(gt)
                    time.sleep(10)

                graceful_time_search(coll, field_name, gt, tti)
        
        csv_filename = str(tti) + ".csv"
        print("writing csv to", csv_filename)

        keys = sorted(rst.keys())
        with open(csv_filename, "w") as f:
           writer = csv.writer(f)
           writer.writerow(keys)
           writer.writerows(zip(*[rst[key] for key in keys]))

    # stop_insert = True
    # t1.join()

    # coll.release()
    # coll.drop()

    print("Done. Don't forget to stop insert program")