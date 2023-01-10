rosbag filter --print="'%s @ %d' % (topic, t.secs)" rania_2022-07-01-11-40-52.bag sample_bag.bag "t.secs > 1656668459 and t.secs<= 1656668469"
