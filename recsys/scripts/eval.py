#!/usr/bin/python
#coding=utf-8

import sys
import re
import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_string('answer', None, 'file location of test data')
gflags.DEFINE_string('guess', None, 'file location model output for test data')
gflags.DEFINE_string('metric', None, 'metric of evaluation')

# super class for eval metric
class Metric(object):
    def eval(self, context):
        pass

class Report(object):
    def __init__(self, score=0.0):
        self.score = score
        pass

class Context(object):
    def __init__(self, answer_path, guess_path):
        self.answer_path = answer_path
        self.guess_path = guess_path


# RMSE for recommend sys
class RMSE(Metric):
    def eval(self, context):
        answer = self.read_from_file(context.answer_path)
        guess = self.read_from_file(context.guess_path)
        keys = set(answer.keys() + guess.keys())
        if not keys: return None
        score = 0.0
        for key in keys:
            x1 = answer.get(key, 0.0)
            x2 = guess.get(key, 0.0)
            score += (x1 - x2) ** 2
        report = Report()
        report.score = (score / len(keys)) ** 0.5
        return report

    def read_from_file(self, path):
        kvs = {}
        for line in open(path).readlines():
            line = line.strip()
            tokens = re.split(r'\s+', line)
            if len(tokens) != 2:
                sys.stderr.write("broken line found: %s\n" % line)
                continue
            key, val = tokens
            kvs[key] = float(val)
        return kvs

if __name__ == "__main__":
    FLAGS(sys.argv)
    context = Context(FLAGS.answer, FLAGS.guess)
    if FLAGS.metric == "rmse":
        rmse = RMSE()
        report = rmse.eval(context)
        print "rmse score: %s" % report.score
    else:
        print "no such a metric named: %s" % metric
