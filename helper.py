from pprint import pprint
""" helper functions """
def names_in(dictionary):
    """ list all names in a dictionary """
    print([name for name,_ in sorted(dictionary.items())])
def names_shape_in(dictionary):
    pprint([(name, val.shape) for name,val in sorted(dictionary.items())])

def debug_show_all_variables():
    global cache, parameters, hyper_parameters
    print("cache: ", end='')
    names_shape_in(cache)
    print("parameters: ", end='')
    names_shape_in(parameters)
    print("hyper_parameters: ", end='')
    names_in(hyper_parameters)
