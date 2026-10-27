from IPython.display import HTML, display

def enable_word_wrap():
  """
  Injects CSS into the Google Colab output frame
  to force long text outputs and reviews to wrap nicely.
  """
  display(HTML('''
  <style>
    /* Force word wrap on standard text/pre outputs */
    pre {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
    /* Force word wrap on active cell execution outputs */
    .output_text pre {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
  </style>
  '''))
