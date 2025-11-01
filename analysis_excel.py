import numpy as np
import matplotlib.pyplot as plt
import collections



total_read_depth,ref_read_depth,alt_read_depth,most_severe_consequence,variant_allele_freq,minor_allele_freq,Ref_Alt = np.genfromtxt(
    r"genomes_data.csv",unpack = True,delimiter = ",",dtype=None,encoding='utf8',skip_header=1)




def graph_maker(column_arr , color , title , x , y , file_name):

    count = collections.Counter(column_arr)


    fig , axs = plt.subplots(figsize=(10,10),facecolor = color)


    categories = list(count.keys())



    values = list(count.values())

    axs.bar(categories , values)
    axs.set_title(title)

    axs.set_xlabel(x)
    plt.xticks(rotation=45)
    axs.set_ylabel(y)

    fig.savefig(file_name)
    plt.show()


    
## we will make graphs for both categorical attributes

graph_maker(Ref_Alt,"beige","Ref-Alt-Count","Ref-Alt-Type","Count","ref_alt.png")
graph_maker(most_severe_consequence,"brown","Variant-Consequence","Consequence-type","Amount","consequence.png")



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
    plt.xticks(rotation=45)
    plt.yscale('log')
    axs.set_ylabel(y)
    fig.savefig(file_name)
    plt.show()



metric_display(total_read_depth,"red","total_read_metrics","metrics","metric_values","tot_read.png")
metric_display(ref_read_depth,"blue","ref_read_metrics","metrics","metric_values","ref_read.png")
metric_display(alt_read_depth,"green","alt_read_metrics","metrics","metric_values","alt_read.png")
metric_display(variant_allele_freq,"orange","variant_allele_metrics","metrics","metric_values","variant_allele.png")
metric_display(minor_allele_freq,"purple","minor_allele_freq","metrics","metric_values","minor_allele.png")


    



    






