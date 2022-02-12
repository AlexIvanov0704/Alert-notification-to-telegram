import pandahouse


class Getch:
    def __init__(self, query, db='simulator'):
        self.connection = {
            'host': 'https://clickhouse.lab.karpov.courses',
            'password': 'password',
            'user': 'user',
            'database': db,
        }
        self.query = query
        self.getchdf

    @property
    def getchdf(self):
        try:
            self.df = pandahouse.read_clickhouse(self.query, connection=self.connection)

        except Exception as err:
            print("\033[31m {}".format(err))
            exit(0)