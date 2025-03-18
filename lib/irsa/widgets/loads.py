import ipywidgets
from IPython.display import display

import irsa.io as io

# _default_keys = [
#     'дата', 'вид_бактерий', 'штамм_бактерий', 
#     'отсечки_по_молекулярной_массе', 'резистентность', 
#     'начальная_концентрация_клеток_в_пробе', 'номер_повтора', 
#     'номер_эксперимента_в_цикле', "комментарий"]

# _cached_attrs = None

def load_spectras(path, dd, options, keys=io.text._default_keys, attrs=None, clear=True):

    if clear:
        dd.clear()    
    
    selector_dict = {}
    selectors = []
    if attrs is None:
        attrs = io.collect_attr_values(path)
    for key in keys:
        vals = attrs[key]
        n_rows = len(vals)
        if n_rows > 5:
            n_rows = 5
        wg = ipywidgets.SelectMultiple(options=vals, description="", rows=n_rows)
        wg.style.font_size="10pt"
        lb = ipywidgets.Label(value=key+":")
        lb.style.font_size="8pt"
        lb.style.font_weight="bold"
        vbox = ipywidgets.VBox((lb,wg))
        selector_dict[key] = wg
        selectors.append(vbox)
        
    box = ipywidgets.Box(selectors)
    box.layout = ipywidgets.Layout(flex_flow="row wrap")
        
    options_button = ipywidgets.Button(description="Select")
    output = ipywidgets.Output()
    
    def onclick_options_button(b, dd=dd, options=options):
        for key,sel in selector_dict.items():
            if sel.value:
                options[key] = sel.value
        dd.update(io.load_spectras(path, options, clear=clear))
        # with output:
        #     print(options)
        #     print(list(dd.keys()))
    
    options_button.on_click(onclick_options_button)
    
    options_widgets = (box, options_button, output)
    display(*options_widgets)
    return dd
