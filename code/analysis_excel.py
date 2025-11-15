import numpy as np
import matplotlib.pyplot as plt
import collections



total_read_depth,ref_read_depth,alt_read_depth,variant_allele_freq,minor_allele_freq,Ref_Alt,most_severe_consequence = np.genfromtxt(
    r"genomes_data_converted.csv",unpack = True,delimiter = ",",dtype=None,encoding='utf8',skip_header=1)




def graph_maker(column_arr , color , title , x , y , file_name):

    


    count = collections.Counter(column_arr)
    categories = list(count.keys())
    if len(categories) == 12:
        consq = ['GA',
'AT',
'GT',
'AC',
'CA',
'GC',
'CG',
'AG',
'TC',
'CT',
'TA',
'TG',]
        for i in range(0,12):
            categories[i] = consq[i]

    if len(categories) == 17:
        consq = ['synonymousvariant',
    'startlost',
    'intergenicvariant',
    'regulatoryregionvariant',
    'upstreamgenevariant',
    'noncodingtranscriptexonvariant',
    'intronvariant',
    'downstreamgenevariant',
    'missensevariant',
    '5primeUTRvariant',
    '3primeUTRvariant',
    'splicedonor5thbasevariant',
    'splicepolypyrimidinetractvariant',
    'splicedonorregionvariant',
    'splicedonorvariant',
    'spliceregionvariant',
    'stoplost']
        for i in range(0,17):
            categories[i] = consq[i]

    


    fig , axs = plt.subplots(figsize=(10,10),facecolor = color)


    



    values = list(count.values())

    axs.bar(categories , values)
    axs.set_title(title)

    axs.set_xlabel(x)
    plt.xticks(rotation=45 , ha='right')
    axs.set_ylabel(y)

    fig.savefig(file_name)
    plt.show()


    
## we will make graphs for both categorical attributes

graph_maker(Ref_Alt,"beige","Ref-Alt-Count","Ref-Alt-Type","Count","finished/ref_alt.png")
graph_maker(most_severe_consequence,"brown","Variant-Consequence","Consequence-type","Amount","finished/consequence.png")



def metric_display(attr , color , title , x , y , file_name):
    mu = np.mean(attr)
    median = np.median(attr)
    std = np.std(attr)
    vx = np.var(attr)

    fig , axs = plt.subplots(figsize = (10,10) , facecolor =  color)

    categories = ["mean","median","standard-deviation","variance"]
    values = [mu,median,std,vx]

    axs.bar(categories,values)
    axs.set_title(title)
    axs.set_xlabel(x)
    plt.xticks(rotation=45,ha="right")
    plt.yscale('log')
    axs.set_ylabel(y)
    fig.savefig(file_name)
    plt.show()



metric_display(total_read_depth,"red","total_read_metrics","metrics","metric_values","finished/tot_read.png")
metric_display(ref_read_depth,"blue","ref_read_metrics","metrics","metric_values","finished/ref_read.png")
metric_display(alt_read_depth,"green","alt_read_metrics","metrics","metric_values","finished/alt_read.png")
metric_display(variant_allele_freq,"orange","variant_allele_metrics","metrics","metric_values","finished/variant_allele.png")
metric_display(minor_allele_freq,"purple","minor_allele_freq","metrics","metric_values","finished/minor_allele.png")


    



    


    



    





