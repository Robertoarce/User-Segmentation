import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def histo(df, col_name='game_id',  title='NONE',symbol= 'M'):   

    if title == 'NONE':
        title = 'Number of Records per ' + str(col_name)
    
    amount = 1
    y_amount= 1
    q_name = ''
    q_symbol= ''
    if symbol == 'M':
        amount = 1000000
        y_amount= 1000
        q_name = '(Thousands)'
        q_symbol= 'M'
    if symbol == 'K':
        amount = 1000
        y_amount= 100
        q_name = '(Hundreds)'
        q_symbol= 'K'


        
        
    df.columns = [col_name, 'count']
    df = df.sort_values('count', ascending=False)
    
    plt.figure(figsize=(14, 7))
    
    ax = sns.barplot(x=col_name, y='count', data=df, palette='viridis')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/y_amount:.0f}{q_symbol}'))
    
    for i, p in enumerate(ax.patches):
        count = df.iloc[i]['count']
        millions = count / amount
        ax.annotate(f'{millions:.2f}{q_symbol}', 
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.xlabel(col_name)
    plt.ylabel(f'Count {q_name}')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()