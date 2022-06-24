import pandas as pd
import os
def Create_Directory_DataFrame(basedir):
    df =pd.DataFrame(columns=['Class','Location'])
    for folder in os.listdir(basedir):
        for Class in os.listdir(basedir+folder+'/'):
            for location in os.listdir(basedir+folder+'/'+Class+'/'):
                df = df.append({'Class':Class,'Location':basedir+folder+'/'+Class+'/'+location},ignore_index=True)
    df = df.sample(frac = 1)
    return df