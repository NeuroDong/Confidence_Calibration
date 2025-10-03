'''
Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H.
'''

from ucimlrepo import fetch_ucirepo

if __name__=="__main__":
    # fetch dataset 
    default_of_credit_card_clients = fetch_ucirepo(id=350) 
    
    # data (as pandas dataframes) 
    X = default_of_credit_card_clients.data.features 
    y = default_of_credit_card_clients.data.targets 
    
    # metadata 
    print(default_of_credit_card_clients.metadata) 
    
    # variable information 
    print(default_of_credit_card_clients.variables) 
