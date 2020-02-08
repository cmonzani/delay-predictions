import gzip
import io
import json
import pprint
import datetime
import math
import random
import hashlib

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


class Dataset_Delay_Prediction_from_list(object):
    def __init__(self, full_dataset_name, datasets_list, data_path, number_of_events, batch_size=64, min_len=5, test_ratio=0.2):
        self.data = []

        self.batch_size = batch_size
        self.number_of_events = number_of_events
        self.dataset_list = datasets_list

        self.min_len = min_len

        self.test_ratio = test_ratio

        self.set_of_users = set([])

        self.max_len = -1
        self.users_dataset = {}
        self.timestamps_of_conversion = {}
        self.timestamps_str_dataset = {}
        self.timestamps_dt_dataset = {}

        print('Generating dataset '+ full_dataset_name)

        for dataset_name in datasets_list:

            print(f'Reading dataset {dataset_name}')
            self.labels_file_path = data_path + 'datasets/' + dataset_name + '/labels.txt.gz'
            self.users_file_path = data_path + 'datasets/' + dataset_name + '/users.txt.gz'
            self.timestamps_file_path = data_path + 'datasets/' + dataset_name + '/timestamps.txt.gz'

            print('Reading labels file...')
            self.labels_file = open(self.labels_file_path, 'rb').read()
            hash_value_for_dataset_name = hashlib.sha256(dataset_name.encode()).hexdigest()[:4]

            with gzip.GzipFile(fileobj=io.BytesIO(self.labels_file), mode='rb') as fo:
                for line in fo:
                    d = json.loads(line)
                    uid = d['uid'] + hash_value_for_dataset_name
                    self.set_of_users.add(uid)
                    self.timestamps_of_conversion[uid] = datetime.datetime.strptime(d["events"][0], time_format)

            self.number_of_users = len(self.set_of_users)

            self.max_event = -1
            print('Reading users file...')
            self.users_file = open(self.users_file_path, 'rb').read()
            with gzip.GzipFile(fileobj=io.BytesIO(self.users_file), mode='rb') as fo:
                for line in fo:
                    d = json.loads(line)
                    uid = d['uid'] + hash_value_for_dataset_name
                    if uid in self.set_of_users:
                        length_of_seq = len(d['events'])
                        if length_of_seq > 2:
                            self.users_dataset[uid] = d['events']
                            self.max_event= max(max(d['events']), self.max_event)

                        else:
                            self.set_of_users.remove(uid)
                            del self.timestamps_of_conversion[uid]
                        if length_of_seq > self.max_len:
                            self.max_len = length_of_seq

            self.number_of_events = self.max_event + 1

            self.timestamps_file = open(self.timestamps_file_path, 'rb').read()

            print('Reading timestamps file...')
            with gzip.GzipFile(fileobj=io.BytesIO(self.timestamps_file), mode='rb') as fo:
                for line in fo:
                    d = json.loads(line)
                    uid = d['uid'] + hash_value_for_dataset_name
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

        self.diff_between_last_event_and_conv = {}

        for uid in self.timestamps_dt_dataset.keys():
            last_event_datetime = self.timestamps_dt_dataset[uid][-1]
            conversion_datetime = self.timestamps_of_conversion[uid]
            diff = (conversion_datetime - last_event_datetime).total_seconds()
            self.diff_between_last_event_and_conv[uid] = diff
            if diff > max_gap:
                max_gap = diff

        self.max_gap = max_gap
        self.constant_C = max_gap / (math.exp(1) - 1)

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

        self.full_seqlen = []

        self.full_values = []
        self.log_values = []

        self.full_features_raw_ts = []
        self.full_features_dt = []
        self.full_features_log = []
        self.full_features_log_dt = []

        for uid in self.set_of_users:
            event_number_list = self.users_dataset[uid]
            timestamps_list = self.timestamps_dt_dataset[uid]
            timestamps_diff = self.timestamps_diff_dataset[uid]
            timestamps_log = self.log_timestamps_diff[uid]

            length_of_seq = len(event_number_list)
            self.full_seqlen.append(length_of_seq)

            val = self.diff_between_last_event_and_conv[uid]
            self.full_values.append([val])
            self.log_values.append([math.log(1 + val / self.constant_C)])

            seq = []
            seq_raw = []
            seq_dt = []
            seq_log = []
            seq_log_dt = []

            aux = [0. for _ in range(self.number_of_events)]
            for idx in range(length_of_seq):
                event = aux.copy()
                event[event_number_list[idx]] = 1

                event_raw = event.copy()
                event_dt = event.copy()
                event_log = event.copy()

                event_raw.append(timestamps_list[idx])
                seq_raw.append(event_raw)

                event_dt.append(timestamps_diff[idx])
                seq_dt.append(event_dt)

                event_log.append(timestamps_log[idx])
                seq_log.append(event_log)

                event_log.append(timestamps_diff[idx])
                seq_log_dt.append(event_log)

            self.full_features_raw_ts.append(seq_raw)
            self.full_features_dt.append(seq_dt)
            self.full_features_log.append(seq_log)
            self.full_features_log_dt.append(seq_log_dt)
        self.training_set_length = int((1.0 - self.test_ratio) * len(self.full_seqlen))
        self.test_set_length = len(self.full_seqlen) - self.training_set_length




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
            event_padding = [0. for _ in range(self.number_of_events + 1)]
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
            val = math.log(1. + val / self.constant_C)

            seqlen.append(length_of_seq)
            y.append([val])

            timestamps = []
            events_list = []

            t_0 = timestamps_list[0]

            for idx in range(self.max_len):

                event = [0. for _ in range(self.number_of_events)]

                if idx < length_of_seq:
                    event[event_number_list[idx]] = 1.
                    t_i = (timestamps_list[idx] - t_0).total_seconds()

                timestamps.append([t_i])
                events_list.append(event)
            x.append(events_list)
            t.append(timestamps)

        return x, t, seqlen, y


class DatasetDelayPredictionStackOverflow(object):
    def __init__(self, data_path, number_of_dimensions, batch_size=64, min_len=10):
        self.data = []
        self.users_file_path = f'{data_path}/event.txt'
        self.timestamps_file_path = f'{data_path}/time.txt'
        self.batch_size = batch_size
        self.number_of_dimensions = number_of_dimensions

        self.min_len = min_len



        self.timestamps_of_conversion = {}

        print('Generating dataset Stack Overflow')

        #Reading event_file:
        print('Reading event file...')
        with open(self.users_file_path) as f:
            content = [line.rstrip() for line in f]
        events_list = [x.split() for x in content]

        self.events_list = [[int(event)-1 for event in events] for events in events_list if len(events) >= min_len]

        self.number_of_events = max(max(self.events_list)) + 1 #To-Do: replace the +1

        print('Reading timestamps file...')
        with open(self.timestamps_file_path) as f:
            content = [line.rstrip() for line in f]
        timestamps_list = [x.split() for x in content]

        timestamps_list = [list_ for list_ in timestamps_list if len(list_) >= min_len]

        self.timestamps_list = [[float(ts) for ts in timestamps] for timestamps in timestamps_list]

        assert len(self.timestamps_list) == len(self.events_list)

        self.number_of_users = len(self.timestamps_list)

        self.timestamps_diffs_list=[]
        print('Generating timestamps differences...')

        for timestamps in self.timestamps_list:
            timestamps_diffs = [0.]
            for i in range(0, len(timestamps)-1):
                timestamps_diffs.append(timestamps[i+1]-timestamps[i])
            self.timestamps_diffs_list.append(timestamps_diffs)

        self.max_gap = max(max(self.timestamps_diffs_list))

        self.constant_C = self.max_gap / (math.exp(1) - 1)

        self.log_timestamps_diffs_list = [[math.log(1 + dt/self.constant_C) for dt in timestamps_diffs] for timestamps_diffs in self.timestamps_diffs_list]

        # print(constant_C)

        print('Generating data for TimeLSTM')
        # dataset for time LSTM (dt as the last dimension of each event)
        self.timestamps_diff = [ts_list[:-1] for ts_list in self.timestamps_diffs_list]
        self.full_values = [[math.log(1 + ts_list[-1]/self.constant_C)] for ts_list in self.timestamps_diffs_list]
        self.next_event = []
        self.full_seqlen = [len(ts_list) - 1 for ts_list in self.timestamps_diffs_list]
        self.full_features_dt = []
        self.full_features_log = []
        self.full_features_log_dt = []

        for i in range(len(self.timestamps_diffs_list)):


            seqlen = self.full_seqlen[i]
            list_of_events = self.events_list[i]
            list_of_timestamps_diff = self.timestamps_diffs_list[i]

            list_of_log_timestamps_diff = self.log_timestamps_diffs_list[i]
            seq = []
            seq_log = []
            seq_log_dt = []
            aux = [0. for _ in range(self.number_of_events)]
            for j in range(seqlen):
                aux_ = aux.copy()
                aux_[list_of_events[j]] = 1.

                aux_log = aux_.copy()
                aux_log.append(list_of_log_timestamps_diff[j])
                seq_log.append(aux_log)

                aux_log.append(list_of_timestamps_diff[j])
                seq_log_dt.append(aux_log)

                aux_.append(list_of_timestamps_diff[j])
                seq.append(aux_)

            j = seqlen + 1
            aux_ = aux.copy()
            aux_[list_of_events[j]] = 1.

            self.next_event.append(aux_)

            self.full_features_dt.append(seq)
            self.full_features_log.append(seq_log)
            self.full_features_log_dt.append(seq_log_dt)

        self.training_set_length = int(4/5 * self.number_of_users)

    def next_for_separated_timestamps_and_events(self):
        # return a batch of data
        # choose users from users_list
        list_ = list([i for i in range(self.training_set_length)])
        users_index = random.sample(list_, k=self.batch_size)

        x = []
        t = []
        seqlen = []
        y = []

        for i in users_index:
            list_of_timestamps =self.timestamps_list[i]
            list_of_events = self.full_features_dt[i]
            timestamps_list = [[0]]
            events_list = [list_of_events[0][:-1]]
            for j in range(1, len(list_of_timestamps)-1):
                timestamps_list.append([list_of_timestamps[j] - list_of_timestamps[0]])
                events_list.append(list_of_events[j][:-1])
            x.append(events_list)
            t.append(timestamps_list)
            seqlen.append(len(events_list))
            y.append([math.log(1 + (list_of_timestamps[-1]-list_of_timestamps[-2])/self.constant_C)])
        padding_vector = [0 for _ in range(self.number_of_events)]
        max_len = max([len(ts_list) for ts_list in t])
        assert max_len == max([len(ev_list) for ev_list in x])

        for i in range(len(x)):
            ev_list = x[i]
            ts_list = t[i]
            if len(ev_list) < max_len:
                for _ in range(max_len - len(ev_list)):
                    ev_list.append(padding_vector)
                    ts_list.append([0])
            x[i] = ev_list
            t[i] = ts_list


        return x, t, seqlen, y

    def test_data_for_separated_timestamps_and_events(self):
        users_index = [i for i in range(self.training_set_length,self.number_of_users)]

        x = []
        t = []
        seqlen = []
        y = []

        for i in users_index:
            list_of_timestamps = self.timestamps_list[i]
            list_of_events = self.full_features_dt[i]
            timestamps_list = [0]
            events_list = [list_of_events[0][:-1]]
            for j in range(1, len(list_of_timestamps) - 1):
                timestamps_list.append(list_of_timestamps[j] - list_of_timestamps[0])
                events_list.append(list_of_events[j][:-1])
            x.append(events_list)
            t.append(timestamps_list)
            seqlen.append(len(events_list))
            y.append([math.log(1 + (list_of_timestamps[-1] - list_of_timestamps[-2]) / self.constant_C)])

        return x, t, seqlen, y
