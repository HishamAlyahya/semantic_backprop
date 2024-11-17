class Question:
    supported_qtypes = ['multiple_choice']

    def __init__(self, question_str: str, output_format: str, qtype: str='', answer=''):
        if qtype == '':
            qtype = 'multiple_choice'
        assert qtype in self.supported_qtypes
        self.question_str = question_str
        self.qtype = qtype
        self.output_format = output_format
        self.answer = answer
        self.id = self.question_str #+ str(random.random())
