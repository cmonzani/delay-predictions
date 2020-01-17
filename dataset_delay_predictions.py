import gzip
import io
import json
import pprint
import datetime
import math
import random

date_format = '%Y/%m/%d'
time_format = '%Y-%m-%dT%H:%M:%S.%fZ'


class Dataset_Delay_Prediction(object):
    def __init__(self, dataset_name, data_path, number_of_events, batch_size=64):
        self.data = []
        self.labels_file_path = data_path + 'datasets/' + dataset_name + '/labels.txt.gz'
        self.users_file_path = data_path + 'datasets/' + dataset_name + '/users.txt.gz'
        self.timestamps_file_path = data_path + 'datasets/' + dataset_name + '/timestamps.txt.gz'
        self.batch_size = batch_size
        self.number_of_events = number_of_events

        self.labels_file = open(self.labels_file_path, 'rb').read()
        self.set_of_users = set([])

        self.max_len = -1

        self.timestamps_of_conversion = {}

        print('Generating dataset '+ dataset_name)

        print('Reading labels file...')

        with gzip.GzipFile(fileobj=io.BytesIO(self.labels_file), mode='rb') as fo:
            for line in fo:
                d = json.loads(line)
                uid = d['uid']
                self.set_of_users.add(uid)
                self.timestamps_of_conversion[uid] = d["events"][0]
                datetime.datetime.strptime(self.timestamps_of_conversion[uid], time_format)

        self.number_of_users = len(self.set_of_users)

        self.users_dataset = {}
        print('Reading users file...')
        self.users_file = open(self.users_file_path, 'rb').read()
        with gzip.GzipFile(fileobj=io.BytesIO(self.users_file), mode='rb') as fo:
            for line in fo:
                d = json.loads(line)
                uid = d['uid']
                if uid in self.set_of_users:
                    length_of_seq = len(d['events'])
                    if length_of_seq > 2:
                        self.users_dataset[uid] = d['events']
                    else:
                        self.set_of_users.remove(uid)
                        del self.timestamps_of_conversion[uid]
                    if length_of_seq > self.max_len:
                        self.max_len = length_of_seq

        self.timestamps_file = open(self.timestamps_file_path, 'rb').read()
        self.timestamps_str_dataset = {}
        self.timestamps_dt_dataset = {}
        print('Reading timestamps file...')
        with gzip.GzipFile(fileobj=io.BytesIO(self.timestamps_file), mode='rb') as fo:
            for line in fo:
                d = json.loads(line)
                uid = d['uid']
                if uid in self.set_of_users:
                    self.timestamps_str_dataset[uid] = d['tm_lists']
                    self.timestamps_dt_dataset[uid] = [datetime.datetime.strptime(timestamp_str, time_format)
                                                       for timestamp_str in d['tm_lists']]

        offset = datetime.timedelta(hours=0)
        tzinfo = datetime.timezone(offset=offset)

        max_gap = -1

        self.timestamps_diff_dataset = {}
        print('Generating timestamps differences...')
        for uid in self.timestamps_str_dataset.keys():
            timestamps_diff = []
            timestamp_dt_list = self.timestamps_dt_dataset[uid].copy()
            # initialize:
            ts_t = timestamp_dt_list.pop(0)
            ts_t_minus_1 = ts_t
            dt = (ts_t - ts_t_minus_1).total_seconds()
            timestamps_diff.append(dt)

            while len(timestamp_dt_list) > 0:
                ts_t = timestamp_dt_list.pop(0)
                dt = (ts_t - ts_t_minus_1).total_seconds()

                timestamps_diff.append(dt)

                ts_t_minus_1 = ts_t
                if dt > max_gap:
                    max_gap = dt
            self.timestamps_diff_dataset[uid] = timestamps_diff

        self.constant_C = max_gap / (math.exp(1) - 1)
        # print(constant_C)
        self.log_timestamps_diff = {}
        for uid in self.timestamps_str_dataset.keys():
            self.log_timestamps_diff[uid] = [math.log(1 + dt / self.constant_C) for dt in
                                             self.timestamps_diff_dataset[uid]]
        if False:
            self.events_with_log_time_appended = {}
            for uid in self.log_timestamps_diff.keys():
                event_number_list = self.users_dataset[uid]
                seq_of_log_timestamps = self.log_timestamps_diff[uid]
                seq_appended = []
                length_of_seq = len(event_number_list)
                for idx in range(self.max_len):
                    event = [0 for _ in range(self.number_of_events)]
                    if idx < length_of_seq:
                        event[event_number_list[idx]] = 1
                        event.append(seq_of_log_timestamps[idx])
                        seq_appended.append(event)
                self.events_with_log_time_appended[uid] = {'seq': seq_appended,
                                                           'seqlen': length_of_seq}

        print('Generating data for TimeLSTM')
        # dataset for time LSTM (dt as the last dimension of each event)
        self.full_features = []
        self.full_seqlen = []
        self.full_values = []

        max_val = -1

        for uid in self.set_of_users:
            event_number_list = self.users_dataset[uid]
            timestamps_list = self.timestamps_dt_dataset[uid]
            length_of_seq = len(event_number_list)
            val = (datetime.datetime.strptime(self.timestamps_of_conversion[uid], time_format)
                   - timestamps_list[-1]).total_seconds()
            if val > max_val:
                max_val = val
            seq_appended = []
            t_im1 = timestamps_list[0]
            for idx in range(length_of_seq):
                event = [0. for _ in range(self.number_of_events)]

                t_i = timestamps_list[idx]
                dt = (t_i - t_im1).total_seconds()
                event[event_number_list[idx]] = 1.
                event.append(dt)
                seq_appended.append(event)
            self.full_features.append(seq_appended)
            self.full_seqlen.append(length_of_seq)
            self.full_values.append([val])

        for idx in range(len(self.full_values)):
            dt = self.full_values[idx][0]
            self.full_values[idx] = [math.log(1 + dt / max_val)]



    def next_for_time_as_a_feature(self):
        # return data for a batch of size:
        # batch_size * max_len * (number_of_different_event + 1)

        users_list = list(self.set_of_users)
        x = []
        seqlen_list = []
        y = []

        ids_of_users = random.choices(users_list, k=self.batch_size)

        for uid in ids_of_users:
            seq = self.events_with_log_time_appended[uid]['seq']
            val = (datetime.datetime.strptime(self.timestamps_of_conversion[uid], time_format)
                   - self.timestamps_dt_dataset[uid][-1]).total_seconds()
            val = math.log(1 + val / self.constant_C)
            seqlen = self.events_with_log_time_appended[uid]['seqlen']
            event_padding = [0 for _ in range(self.number_of_events + 1)]
            seq = seq + [event_padding for _ in range(self.max_len - seqlen)]
            x.append(seq)
            seqlen_list.append(seqlen)
            y.append([val])
        return x, seqlen_list, y

    def next_for_separated_timestamps_and_events(self):
        # return a batch of data
        # choose users from users_list
        users_list = list(self.set_of_users)

        x = []
        t = []
        seqlen = []
        y = []

        ids_of_users = random.choices(users_list, k=self.batch_size)
        for uid in ids_of_users:

            event_number_list = self.users_dataset[uid]

            timestamps_list = self.timestamps_dt_dataset[uid]
            length_of_seq = len(event_number_list)
            val = (datetime.datetime.strptime(self.timestamps_of_conversion[uid], time_format)
                   - timestamps_list[-1]).total_seconds()
            val = math.log(1 + val / self.constant_C)

            seqlen.append(length_of_seq)
            y.append([val])

            timestamps = []
            events_list = []

            t_0 = timestamps_list[0]

            for idx in range(self.max_len):

                event = [0 for _ in range(self.number_of_events)]

                if idx < length_of_seq:
                    event[event_number_list[idx]] = 1
                    t_i = (timestamps_list[idx] - t_0).total_seconds()

                timestamps.append([t_i])
                events_list.append(event)
            x.append(events_list)
            t.append(timestamps)

        return x, t, seqlen, y
