"""
def application(environ, start_response):
    status = '200 OK'
    output = 'Hello World!'
    response_headers = [('Content-type', 'text/plain'),
                        ('Content-Length', str(len(output)))]
    start_response(status, response_headers)
    return [output]
"""
from app import publics
from app2 import get_analyzed_feed

import os
import pickle




def application(env, start_response):
    start_response('200 OK', [('Content-Type','text/html')])

    import ipdb; ipdb.set_trace()

    #return [b"Hello World" + 'hello'.encode('utf-8')]


    #res = [(txt, class_) for txt, class_ in get_analyzed_feed()]
    import ipdb; ipdb.set_trace()

    categories = publics.keys()
    categories = ['<a href="/' + category + '">' + category + '</a>' for category in categories]

    header = ', '.join(categories)


    if os.path.exists('analyzed_feed'):
        analyzed_feed = pickle.load(open('analyzed_feed', 'rb'))
    else:
        analyzed_feed =  list(get_analyzed_feed())
        pickle.dump(analyzed_feed, open('analyzed_feed', 'wb'))


    if env['PATH_INFO'][1:]:
        chosen_category = env['PATH_INFO'][1:].lower()
        res = [('class: ' + j + '<br/>' + i + '<br><br>').encode('utf-8') for i,j in analyzed_feed if j == chosen_category]
    else:
        res = [('class: ' + j + '<br/>' + i + '<br><br>').encode('utf-8') for i,j in analyzed_feed]

    #return ["Hello World"]

    res = header.encode('utf-8') + b''.join(res)
    return [res]

    
    #a = b'<br><br>'.join(res[1])
    """

    #return a
    #return [b"Hello World"]
    """





