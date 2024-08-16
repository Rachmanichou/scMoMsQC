import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

import plotly.graph_objects as go, plotly.express as pxi, plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots

class Summary_Figures:
    
    #Creates the html of every QC Summary plot
    #The metrics, scaled metrics and alert matrices are passed into the constructor. The outlier dataframes will be passed to the functions upon calling.
    def __init__(self, metrics_matrix, scaled_data, alert_matrix):
        self.metrics_matrix = metrics_matrix
        self.scaled_data = scaled_data
        self.alert_matrix = alert_matrix
        
        self.labels = metrics_matrix.index
    
    
    #Create a small summary dataframe and write it as an html file
    def create_summary_table(self, *args):
        summary_table = pd.DataFrame()
        summary_table['Sample ID'] = self.labels
        summary_table['Alerts'] = [sum(row) for i, row in self.alert_matrix.iterrows()]
        for arg in args:
            nb_outliers = np.array( [sum(row) for i, row in arg.iterrows()])

            if not(arg.shape[1] in nb_outliers) and True in arg.columns.str.contains('ATAC') and True in arg.columns.str.contains('GEX'): #The considered OD method operates on each column individually and we are in a multiomics context
                #Display outliers separately for Total, Joint, ATAC and GEX metrics
                nb_ATAC_outliers = np.array( [sum(row) for i, row in arg.filter(regex='ATAC').iterrows()] )
                nb_GEX_outliers = np.array( [sum(row) for i, row in arg.filter(regex='GEX').iterrows()] )
                nb_Joint_outliers = nb_outliers - nb_ATAC_outliers - nb_GEX_outliers

                nb_outliers_txt = [str(o)+' ( ' if o != 0 else str(o) for o in nb_outliers]
                for i in range(0, arg.shape[0]): #Write down
                    nb_outliers_txt[i] += ' '+str(nb_Joint_outliers[i])+' Joint' if nb_Joint_outliers[i] > 0 else ''
                    nb_outliers_txt[i] += ' '+str(nb_ATAC_outliers[i])+' ATAC' if nb_ATAC_outliers[i] > 0 else ''
                    nb_outliers_txt[i] += ' '+str(nb_GEX_outliers[i])+' GEX' if nb_GEX_outliers[i] > 0 else ''
                    nb_outliers_txt[i] += ' )' if nb_outliers_txt[i] != '0' else ''

                summary_table['Outliers with '+arg.OD_method] = nb_outliers_txt
            else:
                summary_table['Outliers with '+arg.OD_method] = ['Outlier' if arg.shape[1] == nb_o else nb_o for nb_o in nb_outliers]

        return summary_table.to_html(classes='table table-striped text-center', justify='center', index=False)
        
        
    #Create a table with all QC values annotated based on whether they are alerts or outliers
    def create_qc_table(self,*args):

        fill_color, text_colors_dfs = self.qc_table_painter(*args)

        table = go.Figure()
        row_height, col_width, base = 30, 100, 208

        #Create a figure for each of the OD methods, beginning with the no OD
        table.add_trace(
            go.Table(
                header=dict(
                    values=list(self.metrics_matrix.columns),
                    fill_color='whitesmoke',
                    align='center',
                    font=dict(color='black', size=12)
                ),
                cells=dict(
                    values=self.metrics_matrix.transpose(),
                    fill_color=fill_color,
                    align='center',
                    font=dict(color='black', size=11)
                ),
                visible=True
            )
        )

        #Creating the buttons which will dictate each of the traces' visibility 
        visibility = [False]*(len(args)+1)
        visibility[0] = True
        buttons = [dict(label="None",
                        method="update",
                        args=[{ "visible":visibility }])]

        for i, arg in enumerate(args):
            table.add_trace(
                    go.Table(
                        header=dict(
                            values=list(self.metrics_matrix.columns),
                            fill_color='whitesmoke',
                            align='center',
                            font=dict(color='black', size=12)
                        ),
                        cells=dict(
                            values=self.metrics_matrix.transpose(),
                            fill_color=fill_color,
                            align='center',
                            font=dict(color=text_colors_dfs[i].transpose(), size=11)
                        ),
                        visible=False
                    )
                )

            visibility = [False]*(len(args)+1)
            visibility[i+1] = True
            buttons.append( dict(label=arg.OD_method,
                            method="update",
                            args=[{ "visible":visibility }]) )

        table.update_layout(
            height=self.metrics_matrix.shape[0]*row_height+base, 
            width=self.metrics_matrix.shape[1]*col_width, 
            showlegend=True,
            #The button system
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.05,
                    y=1.2,
                    buttons=buttons
                )
            ]

        )

        return table.to_html(full_html=False)
    
    
    #Write outliers' values in red and alerts' cells in yellow. Return a list of color dataframes, one for every OD method
    def qc_table_painter(self, *args):
        fill_color = []
        for column in self.metrics_matrix.columns:
            if column != "Sample ID":
                fill_color.append( self.alert_matrix[column].map({True:"bisque",False:"aliceblue"}).to_list() )
            else:
                fill_color.append( ["aliceblue"]*self.alert_matrix.shape[0] )

        text_colors_dfs = []
        for arg in args:
            arg_color_df = pd.DataFrame()
            arg_color_df.OD_method = arg.OD_method
            for column in self.metrics_matrix.columns:
                if column != "Sample ID":
                    arg_color_df[column] = arg[column].map({True:"red",False:"black"})
                else:
                    arg_color_df[column] = ["black"]*self.metrics_matrix.shape[0]
            text_colors_dfs.append(arg_color_df)

        return fill_color, text_colors_dfs
    
    
    #Plotyl violinplot for web output
    def create_violinplot(self, *args):

        fig = go.Figure()
        for column in self.scaled_data.columns:
            #Retrieving the data to be displayed for each point
            alert_array = self.alert_matrix[column].map(lambda a: 'Alert' if a else 'No alert')
            data = np.concatenate(
                                    (np.stack(arrays=[self.labels, alert_array], axis=1), 
                                     np.array([ arg[column].map(lambda o: '<b>Outlying with ' + arg.OD_method + '</b>' if o else 'Inlying with ' + arg.OD_method) for arg in args ]).T),
                                    axis=1)

            fig.add_trace(go.Violin(y=self.scaled_data[column],
                                    name=column,
                                    points='all',
                                    pointpos=0,
                                    customdata=data,
                                    jitter=0.2)        
                            )

        hovertemplate = '<extra></extra>Sample %{customdata[0]} <br>%{customdata[1]} <br>%{customdata[2]}'
        #Create the hover template: the three first columns of the customdata are necessarily alocated (labels, alert, first OD method). Following depends on the number of OD methods used.
        for i in range(3, len(args)+2):
            hovertemplate += '<br>%{customdata['+str(i)+']}'

        fig.update_traces(hovertemplate=hovertemplate,
                          hoveron='points',
                          showlegend=False)

        fig.update_layout(autosize=False,
                          width=2100,
                          height=800
                         )

        return fig.to_html(full_html=False)
    
    #Draw a PCA representation of the scaled data using the three first PCs
    def draw_pca(self, *args):
        scaled_data_array = self.scaled_data.to_numpy()
        X_scaled = PCA(n_components=3).fit(scaled_data_array).transform(scaled_data_array)

        colors = self.painter(*args)

        #Creating a figure for each of the OD methods
        pca = go.Figure()

        #Figure with no outlier annotation
        pca.add_trace(
            go.Scatter3d(
                x=X_scaled[:, 0], #Slicing the array to get each of the first 3 PCs
                y=X_scaled[:, 1],
                z=X_scaled[:, 2],
                text=self.labels,
                mode='markers',
                marker=dict(
                    size=5,
                    color='blue'
                ),
                visible=True
            )
        )
        #Creating the buttons which will dictate each of the traces' visibility 
        visibility = [False]*(len(args)+1)
        visibility[0] = True
        buttons = [dict(label="None",
                        method="update",
                        args=[{ "visible":visibility }])]
        for i, arg in enumerate(args):
            #Figure with OD annotations
            pca.add_trace(
                go.Scatter3d(
                    x=X_scaled[:, 0], #Slicing the array to get each of the first 3 PCs
                    y=X_scaled[:, 1],
                    z=X_scaled[:, 2],
                    text=self.labels,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colors[arg.OD_method]
                    ),
                    visible=False
                )
            )

            visibility = [False]*(len(args)+1)
            visibility[i+1] = True #Element 0 is for no OD method
            buttons.append(dict(label=arg.OD_method,
                                method="update",
                                args=[ { "visible":visibility } ])
            )

        #A button system for choosing which annotation system to display
        pca.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.57,
                    y=1.2,
                    buttons=buttons
                )
            ],
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=0)
        )

        return pca.to_html(full_html=False)
    
    
    #Outliers will be painted in red and inliers in blue
    def painter(self, *args):
        colors = pd.DataFrame()
        for arg in args:
            colors[arg.OD_method] = ['red' if np.any(row) else 'blue' for i, row in arg.iterrows()]
        return colors
    
    
    #Plot a clustermap, a combination of a dendrogram and a heatmap
    def create_clustermap(self, *args):
        #Prepare the data
        htmap_data = self.scaled_data
        htmap_data.index = self.labels

        alert_data = self.alert_matrix.map(lambda a: '<b>Alert</b>' if a else 'No alert')
        alert_data.index = self.labels


        #Make an upper and side dendrogram. Initialize the figure by creating the upper one
        fig = ff.create_dendrogram( htmap_data.T, orientation='bottom', labels = list(htmap_data.columns) )
        for i in range(len(fig['data'])):
            fig['data'][i]['yaxis'] = 'y2'

        dendro_side = ff.create_dendrogram( htmap_data, orientation='right', labels = list(self.labels) )

        for i in range(len(dendro_side['data'])):
            data = dendro_side['data'][i]
            data['xaxis'] = 'x2'
            #Add the side dendrogram's data to the figure
            fig.add_trace(data)

        #Reorder the dataframes based on the hierchical clustering
        htmap_data = htmap_data.reindex(dendro_side['layout']['yaxis']['ticktext'])
        htmap_data = htmap_data[fig['layout']['xaxis']['ticktext']]
        alert_data = alert_data.reindex(dendro_side['layout']['yaxis']['ticktext'])
        alert_data = alert_data[fig['layout']['xaxis']['ticktext']]
        data_tuple = (alert_data,)

        for arg in args:
            reordered_arg = arg
            reordered_arg.index = self.labels
            reordered_arg = reordered_arg.reindex(dendro_side['layout']['yaxis']['ticktext'])
            reordered_arg = reordered_arg[fig['layout']['xaxis']['ticktext']]
            data_tuple += ( reordered_arg.map( lambda o: '<b>Outlying with ' + arg.OD_method + '</b>' if o else 'Inlying with ' + arg.OD_method), )

        #Create the heatmap

        #Create the hover template: the first two columns of the customdata is necessarily alocated (alerts, first OD method). Following depends on the number of OD methods used.
        hovertemplate = '<extra></extra>Sample %{y} <br>%{x} <br>%{customdata[0]}'
        for i in range(2, len(args)+1):
            hovertemplate += '<br>%{customdata['+str(i)+']}'

        heatmap = [
            go.Heatmap(
                x = fig['layout']['xaxis']['tickvals'],
                y = dendro_side['layout']['yaxis']['tickvals'],
                z = htmap_data,
                colorscale = 'Blues',
                showscale = False,
                customdata = np.stack(data_tuple, axis=-1),
                hovertemplate = hovertemplate
            )
        ]

        # Add Heatmap Data to Figure
        for data in heatmap:
            fig.add_trace(data)


        # Edit Layout
        fig.update_layout({'width':1100, 'height':1000,
                                 'showlegend':False, 'hovermode': 'closest',
                                 })
        # Edit xaxis
        fig.update_layout(xaxis={'domain': [.15, 1],
                                          'mirror': False,
                                          'showgrid': False,
                                          'showline': False,
                                          'zeroline': False,
                                          'ticks':""})
        # Edit xaxis2
        fig.update_layout(xaxis2={'domain': [0, .15],
                                           'mirror': False,
                                           'showgrid': False,
                                           'showline': False,
                                           'zeroline': False,
                                           'showticklabels': False,
                                           'ticks':""
                                 })

        # Edit yaxis
        fig.update_layout(yaxis={'domain': [0, .85],
                                          'ticktext': dendro_side['layout']['yaxis']['ticktext'],
                                          'tickvals': dendro_side['layout']['yaxis']['tickvals'],
                                          'ticklabelposition': "outside bottom",
                                          'tickmode': 'array',
                                          'mirror': False,
                                          'showgrid': False,
                                          'showline': False,
                                          'zeroline': False,
                                          'showticklabels': True,
                                          'ticks': ""
                                },
                            plot_bgcolor='rgba(0,0,0,0)')
        # Edit yaxis2
        fig.update_layout(yaxis2={'domain':[.825, .975],
                                           'mirror': False,
                                           'showgrid': False,
                                           'showline': False,
                                           'zeroline': False,
                                           'showticklabels': False,
                                           'ticks':""
                                 })

        fig.update_yaxes(ticktext=dendro_side['layout']['yaxis']['ticktext'])

        return fig.to_html(full_html=False)
