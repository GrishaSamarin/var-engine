from src.data import load_all
from src.var_models import ewma_volatility
from src.var_models import ewma_volatility, garch_volatility

cfg, prices, asset_rets, port_rets = load_all()
vol = ewma_volatility(port_rets)

from src.var_models import parametric_var
import pandas as pd
from src.var_models import garch_evt_var
from src.backtest import count_exceptions
pvar = parametric_var(port_rets)
flat = pd.Series(pvar, index=port_rets.index)
exc = count_exceptions(flat, port_rets)
print("exceptions:", exc.sum())
print("expected:  ", round(len(port_rets) * 0.01, 1))
print("rate:      ", round(exc.mean(), 4))
print(exc[exc].index.year.value_counts().sort_index())

gvar = 2.326 * garch_volatility(port_rets)
gexc = count_exceptions(gvar, port_rets)
print("garch exceptions:", gexc.sum())
print(gexc[gexc].index.year.value_counts().sort_index())

evar = 2.7829 * garch_volatility(port_rets)
eexc = count_exceptions(evar, port_rets)
print("evt exceptions:", eexc.sum())
print(eexc[eexc].index.year.value_counts().sort_index())