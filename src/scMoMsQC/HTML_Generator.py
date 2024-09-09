from yattag import Doc
import pandas as pd
from texts import od_script, intro_text, zscore_text, dixon_text, LOF_text, OD_note_text, qc_text, violinplot_text, pca_text, clustermap_text

class HTML_Generator:
        
    def __init__(self, summary_table, qc, violinplot, pca, clustermap):
        self.summary_table = summary_table
        self.qc = qc
        self.violinplot = violinplot
        self.pca = pca
        self.clustermap = clustermap
        
    
    #Create a HTML file organizing the figures' HTMLs
    def display_summary(self):
        doc, tag, text, line = Doc().ttl()

        doc.asis('<!DOCTYPE html>')

        with tag('html'):
            with tag('head'):
                doc.asis('<meta charset="UTF-8">')
                doc.asis('<meta name="viewport" content="width=device-width, initial-scale=1">')

                #Install Boostrap to display proper dataframes
                doc.asis('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap.min.css">')
                doc.asis('<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/js/bootstrap.min.js"></script>')

                with tag('style'):
                    text('''p,
                            dt {
                                font-weight: bold;
                                } 
                            dl,
                            dd {
                                font-size: 1.5rem;
                                } 
                            dd {
                                margin-bottom: 1em;
                                }
                            @font-face {
                                font-family: "Open Sans", verdana, arial, sans-serif; 
                                src: url(https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300..800;1,300..800&display=swap);
                                } 
                                * {
                                font-family: "Open Sans", verdana, arial, sans-serif;
                                }

                            .button {
                              background-color: white;
                              border-style: solid;
                              border-width: 1px;
                              color: black;
                              padding: 1px;
                              text-align: center;
                              text-decoration: none;
                              display: inline-block;
                              font-size: 12px;
                              font-weight: bold;
                              margin: 2px 4px;
                              cursor: pointer;
                              border-radius: 40%;
                              height: 20px;
                              width: 20px;
                            }

                            .button:active {
                                background-color: #F8F8FF;
                                border-style: inset;
                            }
                                ''')

                doc.asis('<script>'+od_script+' </script>')

            with tag('body', style="background-color:white; margin: 25px 50px;"):          
                line('h1', 'QC Metrics Summary')
                line('br','')
                with tag('div', klass='container', style="margin: 20px 50px;"):
                    with tag('div', klass='row'):
                        with tag('div', klass='col'):
                            text(intro_text)
                            line('br', '')
                            with tag('h2'):
                                text('Outlier Detection ')
                                with tag('button onclick=toggleHelp("odtxt")', klass='button'):
                                    text('?')
                            with tag('div', id='odtxt', klass='title-help-text-container', style="display: none;"):
                                with tag('dl'):
                                    line('dt', 'Z-Score')
                                    line('dd', zscore_text)
                                    line('dt', 'Dixon Q Test')
                                    line('dd', dixon_text)
                                    line('dt', 'Local Outlier Factor (LOF)')
                                    line('dd', LOF_text)
                                    line('dt', 'Nota Bene')
                                    line('dd', OD_note_text)
                            doc.asis(self.summary_table)

                    with tag('div', klass='row'):
                        with tag('div', klass='col'):
                            with tag('h2'):
                                text('QC Metrics Table ')
                                with tag('button onclick=toggleHelp("qctxt")', klass='button'):
                                    text('?')
                            with tag('div', id='qctxt', klass='title-help-text-container', style="display: none;"):
                                line('dt', 'Table')
                                line('dd', qc_text)
                    with tag('div', klass='row'):
                        with tag('div', klass='col'):
                            doc.asis(self.qc)
                    with tag('div', klass='row'):
                        with tag('div', klass='col'):
                            with tag('h2'):
                                text('Violin Plot ')
                                with tag('button onclick=toggleHelp("violintxt")', klass='button'):
                                    text('?')
                            with tag('div', id='violintxt', klass='title-help-text-container', style="display: none;"):
                                line('dt', 'Plot')
                                line('dd', violinplot_text)
                    with tag('div', klass='row'):
                        with tag('div', klass='col'):
                            doc.asis(self.violinplot)
                    with tag('div', klass='row'):
                        with tag('div', klass='col'):
                            with tag('h2'):
                                text('Principal Component Analysis ')
                                with tag('button onclick=toggleHelp("pcatxt")', klass='button'):
                                        text('?')
                            with tag('div', id='pcatxt', klass='title-help-text-container', style="display: none;"):
                                line('dt', 'Plot')
                                line('dd', pca_text)                        
                    with tag('div', klass='row'):
                        with tag('div', klass='col'):
                            doc.asis(self.pca)
                    with tag('div', klass='row'):
                        with tag('div', klass='col'):
                            with tag('h2'):
                                text('Clustermap ')
                                with tag('button onclick=toggleHelp("clustermaptxt")', klass='button'):
                                    text('?')
                            with tag('div', id='clustermaptxt', klass='title-help-text-container', style="display: none;"):
                                line('dt', 'Map')
                                line('dd', clustermap_text)                         
                    with tag('div', klass='row'):
                        with tag('div', klass='col'):
                           doc.asis(self.clustermap)

        return doc.getvalue()  
