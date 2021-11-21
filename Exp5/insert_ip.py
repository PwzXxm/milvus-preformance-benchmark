import argparse
import datetime
import random
import threading
import time
import gc
import numpy as np
import csv
import signal
import sys
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility, Milvus
connections.add_connection(default={"host": "10.255.156.187", "port": "19530"})
connections.connect("default")

TopK = 1
NQ = 1
Nprobe = 128

time_tick_interval = 100
graceful_times = [0, 50, 100, 300, 500, 1000, 5000]
rst = {}


# def time_costing(func):
#     def core(*args):
#         start = time.time()
#         print(func.__name__, "start time: ", start)
#         res = func(*args)
#         end = time.time()
#         print(func.__name__, "end time: ", end)
#         print(func.__name__, "time cost: ", end-start)
#         return res
#     return core


# def create_collection(collection_name, field_name, dim, partition=None, auto_id=True):
#     pk = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=auto_id)
#     field = FieldSchema(name=field_name, dtype=DataType.FLOAT_VECTOR, dim=dim)
#     schema = CollectionSchema(fields=[pk, field], description="example collection")

#     collection = Collection(name=collection_name, schema=schema)
#     return collection


# @time_costing
# def create_index(collection, field_name):
#     default_index = {"index_type": "IVF_FLAT", "params": {"nlist": 1024}, "metric_type": "L2"}
#     collection.create_index(field_name, default_index)
#     # print("Successfully build index")
#     # print(pymilvus.utility.index_building_progress(collection.name))

#     # collection.drop_index()
#     # print("Successfully drop index")


# # @time_costing
# def search(collection, query_entities, field_name, topK, nprobe, guarantee_timestamp=1):
#     search_params = {"metric_type": "L2", "params": {"nprobe": nprobe}}
#     res = collection.search(query_entities, field_name, search_params, limit=topK,
#                             guarantee_timestamp=guarantee_timestamp)


# @time_costing
def insert(collection, entities, partition):
    params = {"timeout": None, "_async":True}
    mr = collection.insert(entities, partition_name=partition, **params)


# def gen_data_and_insert(collection, nb, batch, dim, partition):
#     for i in range(int(nb/batch)):
#         entities = generate_entities(dim, nb)
#         insert(collection, entities, partition)
#         gc.collect()

# def insert_batch(collection, nb, dim, batch, thread_num=1):
#     for i in range(int(nb/batch)):
#         entities = generate_entities(dim, batch)
#         insert(collection, entities, None)
#         gc.collect()

def insert_parallel(collection, partition, dim, batch, speed):
    entities = generate_entities(dim, batch)
    cnt = 0
    over_cnt = 0
    while True:
        global stop_insert
        if stop_insert:
            break
        insert_start = time.time()
        insert(collection, entities, partition)
        insert_dur = time.time() - insert_start
        # print("insert cost: ", insert_dur)
        if speed < insert_dur:
            over_cnt += 1
        time.sleep(max(0, speed-insert_dur))
        cnt += 1

        if cnt != 0 and cnt % 10000 == 0:
            print("[{0}] sent 10000 insert requests".format(datetime.datetime.now().time()), flush=True)

    print("Sent {0} insert requests in total. {1} over the interval {2}ms.".format(cnt, over_cnt, speed))


def generate_entities(dim, nb) -> list:
    # return [{"name": "float_vector", "type": DataType.FLOAT_VECTOR, "values": [[random.random() for _ in range(dim)] for _ in range(nb)]}, {"name": "id", "type": DataType.INT64, "values": [random.randint(1, 9223372036854775807) for _ in range(nb)]}]
    return [[[random.random() for _ in range(dim)] for _ in range(nb)], [random.randint(1, 9223372036854775807) for _ in range(nb)]]
    # return [[random.randint(1, 9223372036854775807) for _ in range(nb)], [[random.random() for _ in range(dim)] for _ in range(nb)]]


# def insert_data_from_file(coll, nb, dim,  vectors_per_file, batch_size):
#     # logger.info("Load npy file: %s end" % file_name)
#     for j in range(nb // vectors_per_file):
#         s = "%05d" % j
#         fname = "binary_" + str(dim) + "d_" + s + ".npy"
#         data = np.load(fname)
#         vectors = data.tolist()
#         insert(coll, vectors)


# def graceful_time_search(coll, field_name, graceful_time, tti):
#     print("warming up", graceful_time)
#     for i in range(10):
#         query_entities = generate_entities(dim, NQ)
#         time.sleep(random.uniform(0, tti/1000.0))
#         search(coll, query_entities, field_name, TopK, Nprobe, 0)

#     print("time tick interval = ", time_tick_interval, "graceful time = ", graceful_time, "start time = ", time.time())

#     total_time = 0
#     for i in range(100):
#         query_entities = generate_entities(dim, NQ)
#         time.sleep(random.uniform(0, tti/1000.0))
#         t0 = time.time()
#         search(coll, query_entities, field_name, TopK, Nprobe, 0)
#         total_time += (time.time() - t0)
#     rst[graceful_time] = total_time/100
#     print("time tick interval = ", time_tick_interval, "graceful time = ", graceful_time, "end time = ", time.time(), flush=True)

def stop_thread_handler(signum, frame):
    print("Signal", signum, "stopping inserting thread")
    global stop_insert
    stop_insert = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="your script description")  # description参数可以用于插入描述脚本用途的信息，可以为空
    parser.add_argument('--speed', '-s', nargs='?', type=float, help='insert speed, :s', default=0.01)
    parser.add_argument('--batch', '-b', nargs='?', type=int, help='insert batch num', default=10)
    args = parser.parse_args()

    batch = args.batch
    speed = args.speed

    collection_name = "deep_100m_96_ip"
    dim = 96

    print("collections: ", utility.list_collections())

    if not utility.has_collection(collection_name):
        raise Exception(collection_name, "does not exist")

    coll = Collection(collection_name)

    partition_name = "cat"
    if utility.has_partition(collection_name, partition_name):
        coll.drop_partition(partition_name)

    coll.create_partition(partition_name)

    print("partitions: ", coll.partitions)

    coll.load()
    time.sleep(10)

    stop_insert = False

    signal.signal(signal.SIGINT, stop_thread_handler)
    signal.signal(signal.SIGTERM, stop_thread_handler)

    t1 = threading.Thread(target=insert_parallel, args=(coll, partition_name, dim, batch, speed))
    t1.start()

    t1.join()

    # coll.release()
    # coll.drop()