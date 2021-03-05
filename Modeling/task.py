#!/usr/bin/env python
import argparse
import pika
import os

parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
args = parser.parse_args()


def put_file_queue(channel, file):
    for rep in range(5):
        files = file + ', ' + str(rep)
        print(files)
        channel.basic_publish(
            exchange='',
            routing_key='task_queue',
            body=files,
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            ))


credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost', credentials=credentials, heartbeat=5))
channel = connection.channel()

channel.exchange_declare(exchange='dlx', exchange_type='direct')

channel.queue_declare(queue='task_queue', durable=True, arguments={"x-dead-letter-exchange": "dlx"})
dl_queue = channel.queue_declare(queue='dl')

channel.queue_bind(exchange='dlx', routing_key='task_queue', queue=dl_queue.method.queue)

for file in os.listdir(args.input_folder):
    put_file_queue(channel, file)

connection.close()
