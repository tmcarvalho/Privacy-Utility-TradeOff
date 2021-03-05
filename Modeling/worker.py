#!/usr/bin/env python
import functools
import os
import threading
import PredictivePerformance
import argparse
import pika
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
parser.add_argument('--output_folder', type=str, help='Output folder', default="./output")
args = parser.parse_args()


def ack_message(ch, delivery_tag, work_sucess):
    if ch.is_open:
        if work_sucess:
            print(" [x] Done")
            ch.basic_ack(delivery_tag)
        else:
            ch.basic_reject(delivery_tag, requeue=False)
            print("[x] Rejected")
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass


def modeling(file, rep):
    # input
    try:
        print(f'{args.input_folder}/{file}' + ' rep:' + str(rep))
        df = pd.read_csv(f'{args.input_folder}/{file}')
        print(len(df))
        res = {}
        X, y = PredictivePerformance.prepare_data(df)
        res = PredictivePerformance.evaluate_model(X, y, res)
    except:
        return False

    # output
    try:
        output_file = f'{args.output_folder}/{file.split(".")[0]}/{file.split(".")[0]}_res{str(rep)}.npy'
        # if not os.path.isdir(output_file):
        #     os.mkdir(output_file)
        # f = open(output_file, 'w')
        # f.write(res)
        np.save(output_file, res)
        return True
    except:
        return False


def do_work(conn, ch, delivery_tag, body):
    msg = [x.strip() for x in body.decode().split(',')]
    work_sucess = modeling(msg[0], msg[1])
    cb = functools.partial(ack_message, ch, delivery_tag, work_sucess)
    conn.add_callback_threadsafe(cb)


def on_message(ch, method_frame, _header_frame, body, args):
    (conn, thrds) = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(conn, ch, delivery_tag, body))
    t.start()
    thrds.append(t)


credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', credentials=credentials, heartbeat=5))
channel = connection.channel()
channel.queue_declare(queue='task_queue', durable=True, arguments={"x-dead-letter-exchange": "dlx"})
print(' [*] Waiting for messages. To exit press CTRL+C')

channel.basic_qos(prefetch_count=1)

threads = []
on_message_callback = functools.partial(on_message, args=(connection, threads))
channel.basic_consume('task_queue', on_message_callback)

try:
    channel.start_consuming()
except KeyboardInterrupt:
    channel.stop_consuming()

# Wait for all to complete
for thread in threads:
    thread.join()

connection.close()
