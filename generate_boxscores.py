# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 22:00:39 2022

@author: bryan
"""
import pandas
from scipy.linalg import cholesky
import shelve
import numpy as np
import scipy.stats
from scipy.stats import norm


#distributions
d = shelve.open("distributions")
m = shelve.open("means")
p = shelve.open("ppfs")
distributions = d['1']
means = m['1']
ppfs = p['1']
d.close()
m.close()
p.close()
"""
for stat in distributions:
    for i in range(1,10000):
        ppfs[stat] = ppfs.setdefault(stat, []) + [distributions[stat].ppf(i/10000)]
#Shelved
distributions["Def TD"] = scipy.stats.burr(4.126208824929675, 0.2600019324466717, loc=-0.499812, scale=1.05178)
distributions["FGA"] = scipy.stats.rice(1.028987322136138, loc=-0.677999, scale=1.64842)
distributions["FGM"] = scipy.stats.genexpon(0.00265365750130161, 1.9159667967714462, 0.0001780388308651802, loc=-0.546454, scale=0.0350111)
distributions["Int"] = scipy.stats.genexpon(0.09536942596966416, 0.6434881211557328, 0.023567275125478647, loc=-0.499587, scale=0.20988)
distributions["Pass Att"] = scipy.stats.chi2(104.24648963866417, loc=-26.8316, scale=0.591238)
distributions["Pass Cmp"] = scipy.stats.johnsonsu(-1.6296795386648544, 5.727656808489991, loc=12.762, scale=33.1208)
distributions["Pass TD"] = scipy.stats.foldnorm(1.4676734359053405, loc=-0.499658, scale=1.37341)
distributions["Pass Yds"] = scipy.stats.gausshyper(21.479688379536555, 150.142220898544, 91.43636723316735, -0.992681535363271, loc=-195.391, scale=1623.77)
distributions["Rush Att"] = scipy.stats.invgauss(0.02864597770522028, loc=-19.2304, scale=1597.75)
distributions["Rush TD"] = scipy.stats.mielke(1.037925024553739, 4.266283846060114, loc=-0.49936, scale=2.48599)
distributions["Rush Yds"] = scipy.stats.exponnorm(1.4519054245461867, loc=72.048, scale=29.9866)
distributions["Sk"] = scipy.stats.genexpon(0.12819671971752097, 3.1725105000509775, 0.124350413042109, loc=-0.498978, scale=1.58884)

for stat in distributions:
    means[stat] = distributions[stat].mean()
"""    
def ppf_(x, stat):
    x_ = max(1, int(round(x,4) *10000))
    x_ = min(9998, x_)
    stat_ = stat
    if "Opp" in stat:
        stat_ = stat[4:] 
    return ppfs[stat_][x_]
 
    

#generate a correlated set of stats
def simulate(c, l, team_mults, home, away):
    inp = norm.rvs(0, 1, size=24)
    out = np.dot(inp, c)
    l_ = l.copy()
    for stat in l:
        l_.append("Opp "+stat)
    box = {}
    for i in range(len(l_)):
        var = out[i]
        stat = l_[i]
        stat_ = stat
        team = home
        if "Opp" in stat:
            stat_ = stat[4:]
            team = away
        box[stat] = max(0, int(round(team_mults[team][stat_]*ppf_(scipy.stats.norm().cdf(var), stat))))
    return box
        
def adjust(stats, target):
    s = stats.copy()
    total = 0
    wonky = False
    for player in stats:
        total += stats[player]
        if stats[player] < 0:
            wonky = True
    if not wonky:       
        mult = target/total
    else:
        mult = 1
    new_total = 0
    adjs = []
    players = []
    for player in stats:
        adj = mult * s[player]
        adjs.append(adj-int(round(adj)))
        s[player] = adj      
        players.append(player)
        s[player] = max(0, int(round(s[player])))
        new_total += s[player]
    adjs_sorted = adjs.copy()
    adjs_sorted.sort()
    adjs_sorted_reversed = adjs_sorted.copy()
    adjs_sorted_reversed.reverse()
    if target-new_total < 0:
        i = 0
        subtractions = 0
        while target + subtractions < new_total:
            player = adjs.index(adjs_sorted[i%len(adjs_sorted)])
            if s[players[player]] > 0:
                s[players[player]] -= 1
                subtractions += 1
            i += 1
    else:
        for i in range(target-new_total):
            s[players[adjs.index(adjs_sorted_reversed[i%len(adjs_sorted_reversed)])]] += 1
    return s

#fantasy points awarded to defenses based on points allowed
def tier(score, scp=False):
    if scp:
        if score <= 6: return 0
        if score <= 13: return -1
        if score <= 20: return -2
        if score <= 27: return -3
        if score <= 34: return -4
        return -5
    else:    
        if score == 0: return 10
        if score <= 6: return 7
        if score <= 13: return 4
        if score <= 20: return 1
        if score <= 27: return 0
        if score <= 34: return -1
        return -4
    
def yds_tier(yards):
    if yards < 300: return 0
    if yards <= 399: return -1
    if yards <= 449: return -3
    if yards <= 499: return -4
    return -5


        
    
    
    

#read data
df = pandas.read_csv("data.csv")

#calculate mean and std of statistics
mean = df.mean()
std = df.std()

#combine home and away stats
l = list(df)
newdf = pandas.DataFrame()
for s in l:
    if "Opp" not in s:
        newdf[s] = pandas.Series.append(df[s], df["Opp "+s])

l = l[:12]

    
#fit distributions
s = shelve.open('fitted')
summary = {}
for stat in l:
   summary[stat] = s[stat]
s.close()

#compute correlation matrix   
df -= mean
df /= std
df=(df-df.mean())/df.std()
cov = df.cov()
c = cholesky(cov)
c_rush = cholesky([[1, .449802], [.449802, 1]])
c_rec = cholesky([[1, .742563, .307137], [.742563, 1, .513421], [.307137, .513421, 1]])

#calculate mean and std of statistics
mean = newdf.mean()
std = newdf.std()
raw_stats = pandas.read_csv("raw_stats_2021_wk1.csv")
raw_stats = raw_stats.fillna(0)
team_dics = {}
team_inputs = {}
for i in range(len(raw_stats)):
    index = raw_stats.loc[i]
    for cat in raw_stats:
        if cat[-2:] != "sd" and isinstance(index[cat],float):
            try:
                team_dics[index["team"]][cat] = team_dics[index["team"]].setdefault(cat,0)+index[cat]
            except:
                team_dics[index["team"]] = {cat: index[cat]}
for team in team_dics:
    team_dics[team]["fgm"] = team_dics[team]["fg_0019"]+team_dics[team]["fg_2029"]+team_dics[team]["fg_3039"]+team_dics[team]["fg_4049"]+team_dics[team]["fg_50"]
    team_inputs[team] = [team_dics[team]["pass_comp"], team_dics[team]["pass_att"], team_dics[team]["pass_yds"], team_dics[team]["pass_tds"], team_dics[team]["pass_int"], 2, team_dics[team]["rush_att"], team_dics[team]["rush_yds"], team_dics[team]["rush_tds"], team_dics[team]["fgm"], team_dics[team]["fgm"]+team_dics[team]["fg_miss"], team_dics[team]["dst_td"]]
team_mults= {}
for team in team_dics:
    team_mults[team] = {}
    for i in range(len(l)):
        stat = l[i]
        team_mults[team][stat] = team_inputs[team][i] / means[stat]

teams = []
s = shelve.open('scenarios')  
s['games'] = [["ATL", "CLE"], ["BAL", "BUF"], ["DAL", "WAS"], ["DET", "SEA"],["HOU", "LAC"] ,["IND", "TEN"], ["NYG", "CHI"], ["PHI", "JAC"], ["PIT", "NYJ"], ["CAR", "ARI"], ["GB", "NE"], ["LVR", "DEN"]]
games = s['games']
defs = {}
rosters = {}
players = {}
qbs = {}
for game in games:
    for t in game:
        rosters[t] = raw_stats.query("(team == @t)").reset_index()
        for i in range(len(rosters[t])):
            players[t] = players.setdefault(t, []) + [dict(rosters[t].loc[i])]
        defs[t] = dict(rosters[t].loc[len(rosters[t])-2])
        qbs[t] = rosters[t].loc[0]["player"]


s['sites'] = {"dk": [], "yahoo": [], "fd": []}
s['scp'] = []
box_scores = {}
for simulations in range(1000):
    pass_yds = {}
    pass_tds = {}
    pass_int = {}
    rush_yds = {}
    rush_tds = {}
    rec = {}    
    rec_yds = {}
    rec_tds = {}
    fantasy_score = {} 
    dk_fantasy_score = {} 
    fd_fantasy_score = {}
    scp_fantasy_score = {}
    def_score = {}
    for game in games:
        home = game[0]
        away = game[1]
        teams.append(home)
        teams.append(away)   
        home_team = rosters[home]
        away_team = rosters[away]
        pass_yds[home] = {}
        pass_tds[home] = {}
        pass_int[home] = {}
        rush_yds[home] = {}
        rush_tds[home] = {}
        rec[home] = {}    
        rec_yds[home] = {}
        rec_tds[home] = {}
        pass_yds[away] = {}
        pass_tds[away] = {}
        pass_int[away] = {}
        rush_yds[away] = {}
        rush_tds[away] = {}
        rec[away] = {}    
        rec_yds[away] = {}
        rec_tds[away] = {}
        home_d = defs[home]
        away_d = defs[away]    
        team_mults[home]["Sk"] = away_d["dst_sacks"] / means['Sk']
        team_mults[away]["Sk"] = home_d["dst_sacks"] / means['Sk']
        #simulations
        b = simulate(c, l, team_mults, home, away)
        box_scores[home+away] = b
        #rush/rec
        for i in range(len(home_team)):     
            index = players[home][i]
            player = index["player"]
            if index["rush_yds"]:
                rvs = norm.rvs(size=2)
                rvs = np.dot(rvs, c_rush)
                rush_yds[home][player] = index["rush_yds"]/means["Rush Yds"]*ppf_(scipy.stats.norm().cdf(rvs[0]), "Rush Yds")
                rush_tds[home][player] = index["rush_tds"]/means["Rush TD"]*ppf_(scipy.stats.norm().cdf(rvs[1]), "Rush TD")
            else: 
                rush_yds[home][player] = 0
                rush_tds[home][player] = 0
            if index["rec"]:
                rvs = norm.rvs(size=3)
                rvs = np.dot(rvs, c_rec)
                rec[home][player] = index["rec"]/means["Pass Cmp"]*ppf_(scipy.stats.norm().cdf(rvs[0]), "Pass Cmp")
                rec_yds[home][player] = index["rec_yds"]/means["Pass Yds"]*ppf_(scipy.stats.norm().cdf(rvs[1]), "Pass Yds")
                rec_tds[home][player] = index["rec_tds"]/means["Pass TD"]*ppf_(scipy.stats.norm().cdf(rvs[2]), "Pass TD")
            else:
                rec[home][player] = 0
                rec_yds[home][player] = 0
                rec_tds[home][player] = 0
                
                          
        for i in range(len(away_team)):         
            index = players[away][i]
            player = index["player"]
            if index["rush_att"]:
                rvs = norm.rvs(size=2)
                rvs = np.dot(rvs, c_rush)
                rush_yds[away][player] = index["rush_yds"]/means["Rush Yds"]*ppf_(scipy.stats.norm().cdf(rvs[0]), "Rush Yds")
                rush_tds[away][player] = index["rush_tds"]/means["Rush TD"]*ppf_(scipy.stats.norm().cdf(rvs[1]), "Rush TD")
            else: 
                rush_yds[away][player] = 0
                rush_tds[away][player] = 0
                
            if index["rec"]:
                rvs = norm.rvs(size=3)
                rvs = np.dot(rvs, c_rec)
                rec[away][player] = index["rec"]/means["Pass Cmp"]*ppf_(scipy.stats.norm().cdf(rvs[0]), "Pass Cmp")
                rec_yds[away][player] =  index["rec_yds"]/means["Pass Yds"]*ppf_(scipy.stats.norm().cdf(rvs[1]), "Pass Yds")
                rec_tds[away][player] =  rec_tds[home][player] = index["rec_tds"]/means["Pass TD"]*ppf_(scipy.stats.norm().cdf(rvs[2]), "Pass TD")
            else:
                rec[away][player] = 0
                rec_yds[away][player] = 0
                rec_tds[away][player] = 0
                
 
        rush_yds[home] = adjust(rush_yds[home], b["Rush Yds"])
        rush_tds[home] = adjust(rush_tds[home], b["Rush TD"])   
        rec[home] = adjust(rec[home], b["Pass Cmp"])
        rec_yds[home] = adjust(rec_yds[home], b["Pass Yds"])
        rec_tds[home] = adjust(rec_tds[home], b["Pass TD"])
        rush_yds[away] = adjust(rush_yds[away], b["Opp Rush Yds"])
        rush_tds[away] = adjust(rush_tds[away], b["Opp Rush TD"])
        rec[away] = adjust(rec[away], b["Opp Pass Cmp"])
        rec_yds[away] = adjust(rec_yds[away], b["Opp Pass Yds"])
        rec_tds[away] = adjust(rec_tds[away], b["Opp Pass TD"])
        #passing
        home_qb = qbs[home]
        away_qb = qbs[away]
        pass_yds[home][home_qb] = b["Pass Yds"]
        pass_yds[away][away_qb] = b["Opp Pass Yds"]
        pass_tds[home][home_qb] = b["Pass TD"]
        pass_tds[away][away_qb] = b["Opp Pass TD"]
        pass_int[home][home_qb] = b["Int"]
        pass_int[away][away_qb] = b["Opp Int"]   
        
        home_score = away_score = 0  
        home_score += b["Pass TD"]*7
        home_score += b["Rush TD"]*7
        home_score += b["FGM"]*3
        home_yards = b["Pass Yds"]+b["Rush Yds"]
        away_score += b["Opp Pass TD"]*7
        away_score += b["Opp Rush TD"]*7
        away_score += b["Opp FGM"]*3
        away_yards = b["Opp Pass Yds"]+b["Opp Rush Yds"]
      
        fantasy_score[home_d["player"]] = b["Opp Sk"] + b["Opp Int"]*2 + b["Def TD"]*6 + tier(away_score)
        fantasy_score[away_d["player"]] = b["Sk"] + b["Int"]*2 + b["Opp Def TD"]*6 + tier(home_score)
        scp_fantasy_score[home_d["player"]] = b["Opp Sk"]*2 + b["Opp Int"]*3 + b["Def TD"]*6 + tier(away_score, True) + yds_tier(away_yards)
        scp_fantasy_score[away_d["player"]] = b["Sk"]*2 + b["Int"]*3 + b["Opp Def TD"]*6 + tier(home_score, True) + yds_tier(home_yards)
        for team in game:
            for player in pass_yds[team]:
                pyds = pass_yds[team][player]
                fantasy_score[player] = fantasy_score.setdefault(player, 0) + pyds/25 + pass_tds[team][player]*4 - pass_int[team][player]
                dk_fantasy_score[player] = dk_fantasy_score.setdefault(player, 0) + pyds/25 + pass_tds[team][player]*4 - pass_int[team][player]
                if pyds >= 300:
                    dk_fantasy_score[player] += 3
                scp_fantasy_score[player] = scp_fantasy_score.setdefault(player, 0) + pyds/20 + pass_tds[team][player]*6 - pass_int[team][player]
       
            for player in rush_yds[team]:
                ruyds = rush_yds[team][player]
                recyds = rec_yds[team][player]
                fantasy_score[player] = fantasy_score.setdefault(player, 0) + ruyds/10 + rush_tds[team][player]*6 + recyds/10 + rec_tds[team][player]*6 + .5*rec[team][player]
                dk_fantasy_score[player] = dk_fantasy_score.setdefault(player, 0) + ruyds/10 + rush_tds[team][player]*6 + recyds/10 + rec_tds[team][player]*6 + rec[team][player]
                if recyds >= 100:
                    dk_fantasy_score[player] += 3
                if ruyds >= 100:
                    dk_fantasy_score[player] += 3
                scp_fantasy_score[player] = scp_fantasy_score.setdefault(player, 0) + ruyds/10 + rush_tds[team][player]*6 + recyds/10 + rec_tds[team][player]*8 + rec[team][player]
               
        
    s['sites']['yahoo'].append(fantasy_score)
    s['sites']['fd'].append(fantasy_score)
    s['sites']['dk'].append(dk_fantasy_score)
    s['scp'] += [scp_fantasy_score]

s.close()


