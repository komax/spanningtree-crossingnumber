import gurobipy as grb


def set_up_model(name):
    new_model = grb.Model(name)
    #new_model.setParam('OutputFlag', False)
    #new_model.setParam('Threads', 4)
    return new_model
