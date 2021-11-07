import time


def write_to_log(filename, msg):
    blacklist = r'\/:*?"<>|'
    assert 1 not in [i in filename for i in blacklist]

    with open(rf"logs\{filename}.txt", "a") as file:
        file.write(msg)


class Logger:

    def __init__(self):
        self.status = {"DIST": "Distancing Violation", "MASK": "No mask violation"}
        self.filename = str(time.ctime(time.time())).replace(':', '-')

    def log_violation(self, stat, dist=None, con_val=None):

        assert stat in self.status

        msg = self.status[stat]
        if stat == "DIST":
            if dist:
                msg += f" within {dist} meters"
        elif stat == "MASK":
            if con_val:
                msg += f" with confidence {round(con_val, 2)}%"

        msg += f" on {time.ctime(time.time())}"

        write_to_log(self.filename, msg + "\n")
        print(msg)
