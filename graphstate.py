#!/usr/bin/python

# parsing state representing a subgraph
# initialized with dependency graph
#
from __future__ import absolute_import
import copy,sys,re
import cPickle
from parser import *
from common.util import *
from constants import *
from common.SpanGraph import SpanGraph
from common.AMRGraph import *
import numpy as np

class ActionError(Exception):
    pass

class ActionTable(dict):
    '''to do'''
    def add_action(self,action_name):
        key =  len(self.keys())+1
        self[key] = action_name

class GraphState(object):
    """
    Starting from dependency graph, each state represents subgraph in parsing process
    Indexed by current node being handled
    """

    sent = None
    #abt_tokens = None
    deptree = None
    action_table = None
    #new_actions = None
    sentID = 0
    gold_graph = None
    model = None
    verbose = None
    num_updates = 0;


    def __init__(self,sigma,A):
        self.sigma = sigma
        self.idx = self.sigma.top()
        self.cidx = None
        self.beta = None
        #self.beta = Buffer(A.nodes[self.idx].children[:]) if self.idx != -1 else None
        #self.cidx = self.beta.top()
        #if self.beta:
        #    self.cidx = self.beta.top()
        #else:
        #    self.cidx = None
        self.A = A
        self.action_history = []

        #self.left_label_set = set([])
        #self._init_atomics()


    @staticmethod
    def init_state(instance,verbose=0):
        depGraph = SpanGraph.init_dep_graph(instance,instance.tokens)
        #depGraph.pre_merge_netag(instance)
        seq = []
        #if instance.sentID == 104:
        #    import pdb
        #    pdb.set_trace()
        for r in sorted(depGraph.multi_roots,reverse=True): seq += depGraph.postorder(root=r)
        #seq = uniqify(seq)
        seq.append(-1)
        sigma = Buffer(seq)
        sigma.push(START_ID)

        GraphState.text = instance.text
        GraphState.sent = instance.tokens
        #GraphState.abt_tokens = {}
        GraphState.gold_graph = instance.gold_graph
        if GraphState.gold_graph: GraphState.gold_graph.abt_node_table = {}
        GraphState.deptree = depGraph
        GraphState.sentID = instance.comment['id'] if instance.comment else GraphState.sentID + 1
        GraphState.verbose = verbose

        if verbose > 1:
            print >> sys.stderr,"Sentence ID:%s, initial sigma:%s" % (GraphState.sentID,sigma)

        return GraphState(sigma,copy.deepcopy(depGraph))

    @staticmethod
    def init_action_table(actions):
        actionTable = ActionTable()
        for act_type,act_str in actions:
            actionTable[act_type] = act_str

        #GraphState.new_actions = set()
        GraphState.action_table = actionTable

    def _init_atomics(self):
        """
        atomics features for the initial state
        """

        # first parent of current node
        sp1 = GraphState.sent[self.A.nodes[self.idx].parents[0]] if self.A.nodes[self.idx].parents else NOT_ASSIGNED
        # immediate left sibling, immediate right sibling and second right sibling
        if sp1 != NOT_ASSIGNED and len(self.A.nodes[sp1['id']].children) > 1:
            children = self.A.nodes[sp1['id']].children
            idx_order = sorted(children).index(self.idx)
            slsb = GraphState.sent[children[idx_order-1]] if idx_order > 0 else NOT_ASSIGNED
            srsb = GraphState.sent[children[idx_order+1]] if idx_order < len(children)-1 else NOT_ASSIGNED
            sr2sb = GraphState.sent[children[idx_order+2]] if idx_order < len(children)-2 else NOT_ASSIGNED
        else:
            slsb = EMPTY
            srsb = EMPTY
            sr2sb = EMPTY

        '''
        # left first parent of current node
        slp1 = GraphState.sent[self.A.nodes[self.idx].parents[0]] if self.A.nodes[self.idx].parents and self.A.nodes[self.idx].parents[0] < self.idx else NOT_ASSIGNED
        # right last child of current child
        brc1 = GraphState.sent[self.deptree.nodes[self.cidx].children[-1]] if self.cidx and self.A.nodes[self.cidx].children and self.A.nodes[self.cidx].children[-1] > self.cidx  else NOT_ASSIGNED
        # left first parent of current child
        blp1 = GraphState.sent[self.A.nodes[self.cidx].parents[0]] if self.cidx and self.A.nodes[self.cidx].parents and self.A.nodes[self.cidx].parents[0] < self.cidx else NOT_ASSIGNED
        '''
        self.atomics = [{'id':tok['id'],
                         'form':tok['form'],
                         'lemma':tok['lemma'],
                         'pos':tok['pos'],
                         'ne':tok['ne'],
                         'nec':tok['nec'],
                         'neamr':tok['neamr'],
                         'amrgeneral':tok['amrgeneral'],
                         """'w2v0':tok['w2v0'],
                         'w2v1':tok['w2v1'],
                         'w2v2':tok['w2v2'],
                         'w2v3':tok['w2v3'],
                         'w2v4':tok['w2v4'],
                         'w2v5':tok['w2v5'],
                         'w2v6':tok['w2v6'],
                         'w2v7':tok['w2v7'],
                         'w2v8':tok['w2v8'],
                         'w2v9':tok['w2v9'],
                         'w2v10':tok['w2v10'],
                         'w2v11':tok['w2v11'],
                         'w2v12':tok['w2v12'],
                         'w2v13':tok['w2v13'],
                         'w2v14':tok['w2v14'],
                         'w2v15':tok['w2v15'],
                         'w2v16':tok['w2v16'],
                         'w2v17':tok['w2v17'],
                         'w2v18':tok['w2v18'],
                         'w2v19':tok['w2v19'],
                         'w2v20':tok['w2v20'],
                         'w2v21':tok['w2v21'],
                         'w2v22':tok['w2v22'],
                         'w2v23':tok['w2v23'],
                         'w2v24':tok['w2v24'],
                         'w2v25':tok['w2v25'],
                         'w2v26':tok['w2v26'],
                         'w2v27':tok['w2v27'],
                         'w2v28':tok['w2v28'],
                         'w2v29':tok['w2v29'],
                         'w2v30':tok['w2v30'],
                         'w2v31':tok['w2v31'],
                         'w2v32':tok['w2v32'],
                         'w2v33':tok['w2v33'],
                         'w2v34':tok['w2v34'],
                         'w2v35':tok['w2v35'],
                         'w2v36':tok['w2v36'],
                         'w2v37':tok['w2v37'],
                         'w2v38':tok['w2v38'],
                         'w2v39':tok['w2v39'],
                         'w2v40':tok['w2v40'],
                         'w2v41':tok['w2v41'],
                         'w2v42':tok['w2v42'],
                         'w2v43':tok['w2v43'],
                         'w2v44':tok['w2v44'],
                         'w2v45':tok['w2v45'],
                         'w2v46':tok['w2v46'],
                         'w2v47':tok['w2v47'],
                         'w2v48':tok['w2v48'],
                         'w2v49':tok['w2v49'],
                         'w2v50':tok['w2v50'],
                         'w2v51':tok['w2v51'],
                         'w2v52':tok['w2v52'],
                         'w2v53':tok['w2v53'],
                         'w2v54':tok['w2v54'],
                         'w2v55':tok['w2v55'],
                         'w2v56':tok['w2v56'],
                         'w2v57':tok['w2v57'],
                         'w2v58':tok['w2v58'],
                         'w2v59':tok['w2v59'],
                         'w2v60':tok['w2v60'],
                         'w2v61':tok['w2v61'],
                         'w2v62':tok['w2v62'],
                         'w2v63':tok['w2v63'],
                         'w2v64':tok['w2v64'],
                         'w2v65':tok['w2v65'],
                         'w2v66':tok['w2v66'],
                         'w2v67':tok['w2v67'],
                         'w2v68':tok['w2v68'],
                         'w2v69':tok['w2v69'],
                         'w2v70':tok['w2v70'],
                         'w2v71':tok['w2v71'],
                         'w2v72':tok['w2v72'],
                         'w2v73':tok['w2v73'],
                         'w2v74':tok['w2v74'],
                         'w2v75':tok['w2v75'],
                         'w2v76':tok['w2v76'],
                         'w2v77':tok['w2v77'],
                         'w2v78':tok['w2v78'],
                         'w2v79':tok['w2v79'],
                         'w2v80':tok['w2v80'],
                         'w2v81':tok['w2v81'],
                         'w2v82':tok['w2v82'],
                         'w2v83':tok['w2v83'],
                         'w2v84':tok['w2v84'],
                         'w2v85':tok['w2v85'],
                         'w2v86':tok['w2v86'],
                         'w2v87':tok['w2v87'],
                         'w2v88':tok['w2v88'],
                         'w2v89':tok['w2v89'],
                         'w2v90':tok['w2v90'],
                         'w2v91':tok['w2v91'],
                         'w2v92':tok['w2v92'],
                         'w2v93':tok['w2v93'],
                         'w2v94':tok['w2v94'],
                         'w2v95':tok['w2v95'],
                         'w2v96':tok['w2v96'],
                         'w2v97':tok['w2v97'],
                         'w2v98':tok['w2v98'],
                         'w2v99':tok['w2v99'],
                         'w2v100':tok['w2v100'],
                         'w2v101':tok['w2v101'],
                         'w2v102':tok['w2v102'],
                         'w2v103':tok['w2v103'],
                         'w2v104':tok['w2v104'],
                         'w2v105':tok['w2v105'],
                         'w2v106':tok['w2v106'],
                         'w2v107':tok['w2v107'],
                         'w2v108':tok['w2v108'],
                         'w2v109':tok['w2v109'],
                         'w2v110':tok['w2v110'],
                         'w2v111':tok['w2v111'],
                         'w2v112':tok['w2v112'],
                         'w2v113':tok['w2v113'],
                         'w2v114':tok['w2v114'],
                         'w2v115':tok['w2v115'],
                         'w2v116':tok['w2v116'],
                         'w2v117':tok['w2v117'],
                         'w2v118':tok['w2v118'],
                         'w2v119':tok['w2v119'],
                         'w2v120':tok['w2v120'],
                         'w2v121':tok['w2v121'],
                         'w2v122':tok['w2v122'],
                         'w2v123':tok['w2v123'],
                         'w2v124':tok['w2v124'],
                         'w2v125':tok['w2v125'],
                         'w2v126':tok['w2v126'],
                         'w2v127':tok['w2v127'],
                         'w2v128':tok['w2v128'],
                         'w2v129':tok['w2v129'],
                         'w2v130':tok['w2v130'],
                         'w2v131':tok['w2v131'],
                         'w2v132':tok['w2v132'],
                         'w2v133':tok['w2v133'],
                         'w2v134':tok['w2v134'],
                         'w2v135':tok['w2v135'],
                         'w2v136':tok['w2v136'],
                         'w2v137':tok['w2v137'],
                         'w2v138':tok['w2v138'],
                         'w2v139':tok['w2v139'],
                         'w2v140':tok['w2v140'],
                         'w2v141':tok['w2v141'],
                         'w2v142':tok['w2v142'],
                         'w2v143':tok['w2v143'],
                         'w2v144':tok['w2v144'],
                         'w2v145':tok['w2v145'],
                         'w2v146':tok['w2v146'],
                         'w2v147':tok['w2v147'],
                         'w2v148':tok['w2v148'],
                         'w2v149':tok['w2v149'],
                         'w2v150':tok['w2v150'],
                         'w2v151':tok['w2v151'],
                         'w2v152':tok['w2v152'],
                         'w2v153':tok['w2v153'],
                         'w2v154':tok['w2v154'],
                         'w2v155':tok['w2v155'],
                         'w2v156':tok['w2v156'],
                         'w2v157':tok['w2v157'],
                         'w2v158':tok['w2v158'],
                         'w2v159':tok['w2v159'],
                         'w2v160':tok['w2v160'],
                         'w2v161':tok['w2v161'],
                         'w2v162':tok['w2v162'],
                         'w2v163':tok['w2v163'],
                         'w2v164':tok['w2v164'],
                         'w2v165':tok['w2v165'],
                         'w2v166':tok['w2v166'],
                         'w2v167':tok['w2v167'],
                         'w2v168':tok['w2v168'],
                         'w2v169':tok['w2v169'],
                         'w2v170':tok['w2v170'],
                         'w2v171':tok['w2v171'],
                         'w2v172':tok['w2v172'],
                         'w2v173':tok['w2v173'],
                         'w2v174':tok['w2v174'],
                         'w2v175':tok['w2v175'],
                         'w2v176':tok['w2v176'],
                         'w2v177':tok['w2v177'],
                         'w2v178':tok['w2v178'],
                         'w2v179':tok['w2v179'],
                         'w2v180':tok['w2v180'],
                         'w2v181':tok['w2v181'],
                         'w2v182':tok['w2v182'],
                         'w2v183':tok['w2v183'],
                         'w2v184':tok['w2v184'],
                         'w2v185':tok['w2v185'],
                         'w2v186':tok['w2v186'],
                         'w2v187':tok['w2v187'],
                         'w2v188':tok['w2v188'],
                         'w2v189':tok['w2v189'],
                         'w2v190':tok['w2v190'],
                         'w2v191':tok['w2v191'],
                         'w2v192':tok['w2v192'],
                         'w2v193':tok['w2v193'],
                         'w2v194':tok['w2v194'],
                         'w2v195':tok['w2v195'],
                         'w2v196':tok['w2v196'],
                         'w2v197':tok['w2v197'],
                         'w2v198':tok['w2v198'],
                         'w2v199':tok['w2v199'],
                         'w2v200':tok['w2v200'],
                         'w2v201':tok['w2v201'],
                         'w2v202':tok['w2v202'],
                         'w2v203':tok['w2v203'],
                         'w2v204':tok['w2v204'],
                         'w2v205':tok['w2v205'],
                         'w2v206':tok['w2v206'],
                         'w2v207':tok['w2v207'],
                         'w2v208':tok['w2v208'],
                         'w2v209':tok['w2v209'],
                         'w2v210':tok['w2v210'],
                         'w2v211':tok['w2v211'],
                         'w2v212':tok['w2v212'],
                         'w2v213':tok['w2v213'],
                         'w2v214':tok['w2v214'],
                         'w2v215':tok['w2v215'],
                         'w2v216':tok['w2v216'],
                         'w2v217':tok['w2v217'],
                         'w2v218':tok['w2v218'],
                         'w2v219':tok['w2v219'],
                         'w2v220':tok['w2v220'],
                         'w2v221':tok['w2v221'],
                         'w2v222':tok['w2v222'],
                         'w2v223':tok['w2v223'],
                         'w2v224':tok['w2v224'],
                         'w2v225':tok['w2v225'],
                         'w2v226':tok['w2v226'],
                         'w2v227':tok['w2v227'],
                         'w2v228':tok['w2v228'],
                         'w2v229':tok['w2v229'],
                         'w2v230':tok['w2v230'],
                         'w2v231':tok['w2v231'],
                         'w2v232':tok['w2v232'],
                         'w2v233':tok['w2v233'],
                         'w2v234':tok['w2v234'],
                         'w2v235':tok['w2v235'],
                         'w2v236':tok['w2v236'],
                         'w2v237':tok['w2v237'],
                         'w2v238':tok['w2v238'],
                         'w2v239':tok['w2v239'],
                         'w2v240':tok['w2v240'],
                         'w2v241':tok['w2v241'],
                         'w2v242':tok['w2v242'],
                         'w2v243':tok['w2v243'],
                         'w2v244':tok['w2v244'],
                         'w2v245':tok['w2v245'],
                         'w2v246':tok['w2v246'],
                         'w2v247':tok['w2v247'],
                         'w2v248':tok['w2v248'],
                         'w2v249':tok['w2v249'],
                         'w2v250':tok['w2v250'],
                         'w2v251':tok['w2v251'],
                         'w2v252':tok['w2v252'],
                         'w2v253':tok['w2v253'],
                         'w2v254':tok['w2v254'],
                         'w2v255':tok['w2v255'],
                         'w2v256':tok['w2v256'],
                         'w2v257':tok['w2v257'],
                         'w2v258':tok['w2v258'],
                         'w2v259':tok['w2v259'],
                         'w2v260':tok['w2v260'],
                         'w2v261':tok['w2v261'],
                         'w2v262':tok['w2v262'],
                         'w2v263':tok['w2v263'],
                         'w2v264':tok['w2v264'],
                         'w2v265':tok['w2v265'],
                         'w2v266':tok['w2v266'],
                         'w2v267':tok['w2v267'],
                         'w2v268':tok['w2v268'],
                         'w2v269':tok['w2v269'],
                         'w2v270':tok['w2v270'],
                         'w2v271':tok['w2v271'],
                         'w2v272':tok['w2v272'],
                         'w2v273':tok['w2v273'],
                         'w2v274':tok['w2v274'],
                         'w2v275':tok['w2v275'],
                         'w2v276':tok['w2v276'],
                         'w2v277':tok['w2v277'],
                         'w2v278':tok['w2v278'],
                         'w2v279':tok['w2v279'],
                         'w2v280':tok['w2v280'],
                         'w2v281':tok['w2v281'],
                         'w2v282':tok['w2v282'],
                         'w2v283':tok['w2v283'],
                         'w2v284':tok['w2v284'],
                         'w2v285':tok['w2v285'],
                         'w2v286':tok['w2v286'],
                         'w2v287':tok['w2v287'],
                         'w2v288':tok['w2v288'],
                         'w2v289':tok['w2v289'],
                         'w2v290':tok['w2v290'],
                         'w2v291':tok['w2v291'],
                         'w2v292':tok['w2v292'],
                         'w2v293':tok['w2v293'],
                         'w2v294':tok['w2v294'],
                         'w2v295':tok['w2v295'],
                         'w2v296':tok['w2v296'],
                         'w2v297':tok['w2v297'],
                         'w2v298':tok['w2v298'],
                         'w2v299':tok['w2v299'],
                         'life':tok['life'],
                         'spell':tok['spell'],
                         'store0supply':tok['store0supply'],
                         'knowledge':tok['knowledge'],
                         'other0sports':tok['other0sports'],
                         'appearance':tok['appearance'],
                         'age':tok['age'],
                         'uncertainty':tok['uncertainty'],
                         'borrowing':tok['borrowing'],
                         'religions0cults0sects':tok['religions0cults0sects'],
                         'visibility':tok['visibility'],
                         'custom0habit':tok['custom0habit'],
                         'environment':tok['environment'],
                         'accounts':tok['accounts'],
                         'fragrance':tok['fragrance'],
                         'demand':tok['demand'],
                         'notch':tok['notch'],
                         'disproof':tok['disproof'],
                         'public0spirit':tok['public0spirit'],
                         'pity':tok['pity'],
                         'prosperity':tok['prosperity'],
                         'direction':tok['direction'],
                         'convolution':tok['convolution'],
                         'lowness':tok['lowness'],
                         'artist':tok['artist'],
                         'dance':tok['dance'],
                         'leisure':tok['leisure'],
                         'incombustibility':tok['incombustibility'],
                         'quiescence':tok['quiescence'],
                         'engineering':tok['engineering'],
                         'printing':tok['printing'],
                         'behavior':tok['behavior'],
                         'action':tok['action'],
                         'compact':tok['compact'],
                         'sequel':tok['sequel'],
                         'rejection':tok['rejection'],
                         'necessity':tok['necessity'],
                         'unwillingness':tok['unwillingness'],
                         'public0speaking':tok['public0speaking'],
                         'unrelatedness':tok['unrelatedness'],
                         'peace':tok['peace'],
                         'period':tok['period'],
                         'prediction':tok['prediction'],
                         'resignation0retirement':tok['resignation0retirement'],
                         'the0people':tok['the0people'],
                         'youngster':tok['youngster'],
                         'stridency':tok['stridency'],
                         'counteraction':tok['counteraction'],
                         'organic0matter':tok['organic0matter'],
                         'misbehavior':tok['misbehavior'],
                         'unpleasantness':tok['unpleasantness'],
                         'badness':tok['badness'],
                         'rashness':tok['rashness'],
                         'tribunal':tok['tribunal'],
                         'biology':tok['biology'],
                         'eagerness':tok['eagerness'],
                         'wealth':tok['wealth'],
                         'formality':tok['formality'],
                         'money':tok['money'],
                         'sharpness':tok['sharpness'],
                         'will':tok['will'],
                         'courage':tok['courage'],
                         'invisibility':tok['invisibility'],
                         'nonconformity':tok['nonconformity'],
                         'property':tok['property'],
                         'simultaneity':tok['simultaneity'],
                         'filament':tok['filament'],
                         'unpreparedness':tok['unpreparedness'],
                         'accord':tok['accord'],
                         'sphericity0rotundity':tok['sphericity0rotundity'],
                         'discovery':tok['discovery'],
                         'productiveness':tok['productiveness'],
                         'irresolution':tok['irresolution'],
                         'space0travel':tok['space0travel'],
                         'caprice':tok['caprice'],
                         'bisection':tok['bisection'],
                         'commission':tok['commission'],
                         'electronics':tok['electronics'],
                         'seclusion':tok['seclusion'],
                         'honor':tok['honor'],
                         'market':tok['market'],
                         'nonexistence':tok['nonexistence'],
                         'conciseness':tok['conciseness'],
                         'density':tok['density'],
                         'piety':tok['piety'],
                         'probability':tok['probability'],
                         'the0ministry':tok['the0ministry'],
                         'interval':tok['interval'],
                         'materiality':tok['materiality'],
                         'religious0buildings':tok['religious0buildings'],
                         'misteaching':tok['misteaching'],
                         'location':tok['location'],
                         'slowness':tok['slowness'],
                         'mediocrity':tok['mediocrity'],
                         'security':tok['security'],
                         'furrow':tok['furrow'],
                         'illusion':tok['illusion'],
                         'nativeness':tok['nativeness'],
                         'no0qualifications':tok['no0qualifications'],
                         'hospitality0welcome':tok['hospitality0welcome'],
                         'cleanness':tok['cleanness'],
                         'student':tok['student'],
                         'solemnity':tok['solemnity'],
                         'transience':tok['transience'],
                         'ingratitude':tok['ingratitude'],
                         'middle':tok['middle'],
                         'excretion':tok['excretion'],
                         'protection':tok['protection'],
                         'remainder':tok['remainder'],
                         'vehicle':tok['vehicle'],
                         'exclusion':tok['exclusion'],
                         'resonance':tok['resonance'],
                         'unsavoriness':tok['unsavoriness'],
                         'adversity':tok['adversity'],
                         'preparation':tok['preparation'],
                         'cardplaying':tok['cardplaying'],
                         'giving':tok['giving'],
                         'restraint':tok['restraint'],
                         'morning0noon':tok['morning0noon'],
                         'clothing0materials':tok['clothing0materials'],
                         'carefulness':tok['carefulness'],
                         'enclosure':tok['enclosure'],
                         'improbity':tok['improbity'],
                         'pursuit':tok['pursuit'],
                         'impulse':tok['impulse'],
                         'topic':tok['topic'],
                         'affectation':tok['affectation'],
                         'rocketry0missilery':tok['rocketry0missilery'],
                         'the0country':tok['the0country'],
                         'leading':tok['leading'],
                         'the0future':tok['the0future'],
                         'provision0equipment':tok['provision0equipment'],
                         'generality':tok['generality'],
                         'disillusionment':tok['disillusionment'],
                         'error':tok['error'],
                         'politics':tok['politics'],
                         'cheapness':tok['cheapness'],
                         'refreshment':tok['refreshment'],
                         'automation':tok['automation'],
                         'hair':tok['hair'],
                         'measurement0of0time':tok['measurement0of0time'],
                         'evening0night':tok['evening0night'],
                         'repulsion':tok['repulsion'],
                         'legality':tok['legality'],
                         'cooperation':tok['cooperation'],
                         'promotion':tok['promotion'],
                         'mariner':tok['mariner'],
                         'sculpture':tok['sculpture'],
                         'rear':tok['rear'],
                         'stench':tok['stench'],
                         'plants':tok['plants'],
                         'soliloquy':tok['soliloquy'],
                         'unastonishment':tok['unastonishment'],
                         'unsubstantiality':tok['unsubstantiality'],
                         'eccentricity':tok['eccentricity'],
                         'financial0credit':tok['financial0credit'],
                         'compensation':tok['compensation'],
                         'rarity':tok['rarity'],
                         'attraction':tok['attraction'],
                         'restoration':tok['restoration'],
                         'perpetuity':tok['perpetuity'],
                         'pulpiness':tok['pulpiness'],
                         'accompaniment':tok['accompaniment'],
                         'doubleness':tok['doubleness'],
                         'stock0market':tok['stock0market'],
                         'friend':tok['friend'],
                         'literature':tok['literature'],
                         'curiosity':tok['curiosity'],
                         'weight':tok['weight'],
                         'presence':tok['presence'],
                         'justice':tok['justice'],
                         'plain':tok['plain'],
                         'defeat':tok['defeat'],
                         'haste':tok['haste'],
                         'demotion0deposal':tok['demotion0deposal'],
                         'skiing':tok['skiing'],
                         'cessation':tok['cessation'],
                         'incuriosity':tok['incuriosity'],
                         'semiliquidity':tok['semiliquidity'],
                         'ornamentation':tok['ornamentation'],
                         'nonimitation':tok['nonimitation'],
                         'loss':tok['loss'],
                         'inelegance':tok['inelegance'],
                         'rejoicing':tok['rejoicing'],
                         'communication':tok['communication'],
                         'tools0machinery':tok['tools0machinery'],
                         'securities':tok['securities'],
                         'wrong':tok['wrong'],
                         'concavity':tok['concavity'],
                         'expenditure':tok['expenditure'],
                         'innocence':tok['innocence'],
                         'sophistry':tok['sophistry'],
                         'power0potency':tok['power0potency'],
                         'dullness':tok['dullness'],
                         'nonaccomplishment':tok['nonaccomplishment'],
                         'structure':tok['structure'],
                         'virtue':tok['virtue'],
                         'savoriness':tok['savoriness'],
                         'deception':tok['deception'],
                         'publication':tok['publication'],
                         'intrusion':tok['intrusion'],
                         'danger':tok['danger'],
                         'importance':tok['importance'],
                         'evolution':tok['evolution'],
                         'top':tok['top'],
                         'legal0action':tok['legal0action'],
                         'disagreement':tok['disagreement'],
                         'elasticity':tok['elasticity'],
                         'texture':tok['texture'],
                         'possibility':tok['possibility'],
                         'quadruplication':tok['quadruplication'],
                         'the0past':tok['the0past'],
                         'secretion':tok['secretion'],
                         'weaving':tok['weaving'],
                         'judgment':tok['judgment'],
                         'optical0instruments':tok['optical0instruments'],
                         'rescue':tok['rescue'],
                         'desire':tok['desire'],
                         'treatise':tok['treatise'],
                         'vulgarity':tok['vulgarity'],
                         'discontinuity':tok['discontinuity'],
                         'united0nations0international0organizations':tok['united0nations0international0organizations'],
                         'psychology0psychotherapy':tok['psychology0psychotherapy'],
                         'displacement':tok['displacement'],
                         'insertion':tok['insertion'],
                         'littleness':tok['littleness'],
                         'wonder':tok['wonder'],
                         'numerousness':tok['numerousness'],
                         'abnormality':tok['abnormality'],
                         'infinity':tok['infinity'],
                         'kindness0benevolence':tok['kindness0benevolence'],
                         'form0':tok['form0'],
                         'sequence':tok['sequence'],
                         'undertaking':tok['undertaking'],
                         'wise0saying':tok['wise0saying'],
                         'costlessness':tok['costlessness'],
                         'rain':tok['rain'],
                         'wakefulness':tok['wakefulness'],
                         'bane':tok['bane'],
                         'posterity':tok['posterity'],
                         'heat':tok['heat'],
                         'music':tok['music'],
                         'degree':tok['degree'],
                         'rotation':tok['rotation'],
                         'liquidity':tok['liquidity'],
                         'sameness':tok['sameness'],
                         'defective0vision':tok['defective0vision'],
                         'product':tok['product'],
                         'misrepresentation':tok['misrepresentation'],
                         'boasting':tok['boasting'],
                         'horizontalness':tok['horizontalness'],
                         'associate':tok['associate'],
                         'hardness0rigidity':tok['hardness0rigidity'],
                         'softness0pliancy':tok['softness0pliancy'],
                         'concurrence':tok['concurrence'],
                         'radar0radiolocators':tok['radar0radiolocators'],
                         'unhealthfulness':tok['unhealthfulness'],
                         'liberation':tok['liberation'],
                         'complexity':tok['complexity'],
                         'defense':tok['defense'],
                         'health':tok['health'],
                         'interment':tok['interment'],
                         'pretext':tok['pretext'],
                         'deity':tok['deity'],
                         'redness':tok['redness'],
                         'illicit0business':tok['illicit0business'],
                         'inattention':tok['inattention'],
                         'greenness':tok['greenness'],
                         'subjection':tok['subjection'],
                         'fastidiousness':tok['fastidiousness'],
                         'blemish':tok['blemish'],
                         'command':tok['command'],
                         'unaccustomedness':tok['unaccustomedness'],
                         'deafness':tok['deafness'],
                         'certainty':tok['certainty'],
                         'ignorance':tok['ignorance'],
                         'normality':tok['normality'],
                         'wit0humor':tok['wit0humor'],
                         'the0body':tok['the0body'],
                         'parallelism':tok['parallelism'],
                         'truth':tok['truth'],
                         'contrariety':tok['contrariety'],
                         'inexpectation':tok['inexpectation'],
                         'architecture0design':tok['architecture0design'],
                         'pushing0throwing':tok['pushing0throwing'],
                         'previousness':tok['previousness'],
                         'feeling':tok['feeling'],
                         'equality':tok['equality'],
                         'toughness':tok['toughness'],
                         'offer':tok['offer'],
                         'permanence':tok['permanence'],
                         'influence':tok['influence'],
                         'sensation':tok['sensation'],
                         'radio':tok['radio'],
                         'good0person':tok['good0person'],
                         'fatigue':tok['fatigue'],
                         'hockey':tok['hockey'],
                         'falseness':tok['falseness'],
                         'answer':tok['answer'],
                         'earliness':tok['earliness'],
                         'inexpedience':tok['inexpedience'],
                         'soccer':tok['soccer'],
                         'conformity':tok['conformity'],
                         'continuance':tok['continuance'],
                         'inactivity':tok['inactivity'],
                         'pitilessness':tok['pitilessness'],
                         'evidence0proof':tok['evidence0proof'],
                         'disobedience':tok['disobedience'],
                         'precursor':tok['precursor'],
                         'energy':tok['energy'],
                         'manifestation':tok['manifestation'],
                         'harmonics0musical0elements':tok['harmonics0musical0elements'],
                         'director':tok['director'],
                         'signs0indicators':tok['signs0indicators'],
                         'blackness':tok['blackness'],
                         'learning':tok['learning'],
                         'brownness':tok['brownness'],
                         'cause':tok['cause'],
                         'killing':tok['killing'],
                         'aristocracy0nobility0gentry':tok['aristocracy0nobility0gentry'],
                         'unsociability':tok['unsociability'],
                         'prodigality':tok['prodigality'],
                         'illegality':tok['illegality'],
                         'impatience':tok['impatience'],
                         'triplication':tok['triplication'],
                         'addition':tok['addition'],
                         'air0weather':tok['air0weather'],
                         'formlessness':tok['formlessness'],
                         'celibacy':tok['celibacy'],
                         'sociability':tok['sociability'],
                         'sanity':tok['sanity'],
                         'attack':tok['attack'],
                         'musical0instruments':tok['musical0instruments'],
                         'transparency':tok['transparency'],
                         'secrecy':tok['secrecy'],
                         'loudness':tok['loudness'],
                         'emergence':tok['emergence'],
                         'parsimony':tok['parsimony'],
                         'contraction':tok['contraction'],
                         'interim':tok['interim'],
                         'observance':tok['observance'],
                         'regret':tok['regret'],
                         'symmetry':tok['symmetry'],
                         'similarity':tok['similarity'],
                         'country':tok['country'],
                         'show0business0theater':tok['show0business0theater'],
                         'sufficiency':tok['sufficiency'],
                         'retaliation':tok['retaliation'],
                         'four':tok['four'],
                         'periodical':tok['periodical'],
                         'changeableness':tok['changeableness'],
                         'following':tok['following'],
                         'specter':tok['specter'],
                         'cold':tok['cold'],
                         'ostentation':tok['ostentation'],
                         'hate':tok['hate'],
                         'contempt':tok['contempt'],
                         'sensuality':tok['sensuality'],
                         'stability':tok['stability'],
                         'ill0humor':tok['ill0humor'],
                         'expedience':tok['expedience'],
                         'convexity0protuberance':tok['convexity0protuberance'],
                         'route0path':tok['route0path'],
                         'circumstance':tok['circumstance'],
                         'untimeliness':tok['untimeliness'],
                         'whiteness':tok['whiteness'],
                         'insufficiency':tok['insufficiency'],
                         'experiment':tok['experiment'],
                         'fear0fearfulness':tok['fear0fearfulness'],
                         'chastity':tok['chastity'],
                         'measurement':tok['measurement'],
                         'convergence':tok['convergence'],
                         'motivation0inducement':tok['motivation0inducement'],
                         'legislature0government0organization':tok['legislature0government0organization'],
                         'unregretfulness':tok['unregretfulness'],
                         'title':tok['title'],
                         'plunge':tok['plunge'],
                         'end':tok['end'],
                         'classification':tok['classification'],
                         'permission':tok['permission'],
                         'odor':tok['odor'],
                         'assent':tok['assent'],
                         'obedience':tok['obedience'],
                         'roughness':tok['roughness'],
                         'youth':tok['youth'],
                         'neutrality':tok['neutrality'],
                         'shade':tok['shade'],
                         'worker0doer':tok['worker0doer'],
                         'underestimation':tok['underestimation'],
                         'insolence':tok['insolence'],
                         'intuition0instinct':tok['intuition0instinct'],
                         'misinterpretation':tok['misinterpretation'],
                         'intention':tok['intention'],
                         'insignia':tok['insignia'],
                         'agriculture':tok['agriculture'],
                         'deputy0agent':tok['deputy0agent'],
                         'diffuseness':tok['diffuseness'],
                         'agreement':tok['agreement'],
                         'elegance':tok['elegance'],
                         'dispersion':tok['dispersion'],
                         'leap':tok['leap'],
                         'vapor0gas':tok['vapor0gas'],
                         'time':tok['time'],
                         'prose':tok['prose'],
                         'transfer0of0property0or0right':tok['transfer0of0property0or0right'],
                         'celebration':tok['celebration'],
                         'tedium':tok['tedium'],
                         'discontent':tok['discontent'],
                         'curse':tok['curse'],
                         'congratulation':tok['congratulation'],
                         'sourness':tok['sourness'],
                         'probity':tok['probity'],
                         'substitution':tok['substitution'],
                         'idea':tok['idea'],
                         'improvement':tok['improvement'],
                         'fold':tok['fold'],
                         'oneness':tok['oneness'],
                         'ecclesiastical0attire':tok['ecclesiastical0attire'],
                         'baseball':tok['baseball'],
                         'ambiguity':tok['ambiguity'],
                         'leverage0purchase':tok['leverage0purchase'],
                         'traveler':tok['traveler'],
                         'artlessness':tok['artlessness'],
                         'sex':tok['sex'],
                         'neglect':tok['neglect'],
                         'mechanics':tok['mechanics'],
                         'land':tok['land'],
                         'word':tok['word'],
                         'tobacco':tok['tobacco'],
                         'reversion':tok['reversion'],
                         'timeliness':tok['timeliness'],
                         'refusal':tok['refusal'],
                         'inferiority':tok['inferiority'],
                         'bad0person':tok['bad0person'],
                         'substance0abuse':tok['substance0abuse'],
                         'increase':tok['increase'],
                         'left0side':tok['left0side'],
                         'criticism0of0the0arts':tok['criticism0of0the0arts'],
                         'price0fee':tok['price0fee'],
                         'unsanctity':tok['unsanctity'],
                         'master':tok['master'],
                         'uncleanness':tok['uncleanness'],
                         'insignificance':tok['insignificance'],
                         'advice':tok['advice'],
                         'association':tok['association'],
                         'misanthropy':tok['misanthropy'],
                         'revolution':tok['revolution'],
                         'fitness0exercise':tok['fitness0exercise'],
                         'success':tok['success'],
                         'resentment0anger':tok['resentment0anger'],
                         'dupe':tok['dupe'],
                         'departure':tok['departure'],
                         'adjunct':tok['adjunct'],
                         'difficulty':tok['difficulty'],
                         'disapproval':tok['disapproval'],
                         'separation':tok['separation'],
                         'activity':tok['activity'],
                         'difference':tok['difference'],
                         'regression':tok['regression'],
                         'channel':tok['channel'],
                         'contraposition':tok['contraposition'],
                         'circularity':tok['circularity'],
                         'imposition':tok['imposition'],
                         'enmity':tok['enmity'],
                         'beginning':tok['beginning'],
                         'lending':tok['lending'],
                         'representation0description':tok['representation0description'],
                         'list':tok['list'],
                         'businessman0merchant':tok['businessman0merchant'],
                         'centrality':tok['centrality'],
                         'excitement':tok['excitement'],
                         'taste':tok['taste'],
                         'marriage':tok['marriage'],
                         'premonition':tok['premonition'],
                         'moisture':tok['moisture'],
                         'pleasure':tok['pleasure'],
                         'restitution':tok['restitution'],
                         'unionism0labor0union':tok['unionism0labor0union'],
                         'gratitude':tok['gratitude'],
                         'interpretation':tok['interpretation'],
                         'contents':tok['contents'],
                         'purpleness':tok['purpleness'],
                         'sewing':tok['sewing'],
                         'entrance':tok['entrance'],
                         'sadness':tok['sadness'],
                         'approval':tok['approval'],
                         'duration':tok['duration'],
                         'ascent':tok['ascent'],
                         'idolatry':tok['idolatry'],
                         'acquisition':tok['acquisition'],
                         'abode0habitat':tok['abode0habitat'],
                         'animal0sounds':tok['animal0sounds'],
                         'orangeness':tok['orangeness'],
                         'basketball':tok['basketball'],
                         'animals0insects':tok['animals0insects'],
                         'immateriality':tok['immateriality'],
                         'conversation':tok['conversation'],
                         'lamentation':tok['lamentation'],
                         'rest0repose':tok['rest0repose'],
                         'leniency':tok['leniency'],
                         'cloud':tok['cloud'],
                         'the0laity':tok['the0laity'],
                         'consumption':tok['consumption'],
                         'laxness':tok['laxness'],
                         'condemnation':tok['condemnation'],
                         'abridgment':tok['abridgment'],
                         'discrimination':tok['discrimination'],
                         'height':tok['height'],
                         'recession':tok['recession'],
                         'unproductiveness':tok['unproductiveness'],
                         'relation':tok['relation'],
                         'qualification':tok['qualification'],
                         'negation0denial':tok['negation0denial'],
                         'bodily0development':tok['bodily0development'],
                         'mathematics':tok['mathematics'],
                         'intemperance':tok['intemperance'],
                         'scripture':tok['scripture'],
                         'skill':tok['skill'],
                         'occultism':tok['occultism'],
                         'right':tok['right'],
                         'lake0pool':tok['lake0pool'],
                         'vice':tok['vice'],
                         'chemistry0chemicals':tok['chemistry0chemicals'],
                         'disorder':tok['disorder'],
                         'informality':tok['informality'],
                         'disparagement':tok['disparagement'],
                         'angularity':tok['angularity'],
                         'unselfishness':tok['unselfishness'],
                         'meaning':tok['meaning'],
                         'sleep':tok['sleep'],
                         'grayness':tok['grayness'],
                         'finance0investment':tok['finance0investment'],
                         'fuel':tok['fuel'],
                         'contentment':tok['contentment'],
                         'nonuniformity':tok['nonuniformity'],
                         'improbability':tok['improbability'],
                         'expectation':tok['expectation'],
                         'possession':tok['possession'],
                         'expansion0growth':tok['expansion0growth'],
                         'ridicule':tok['ridicule'],
                         'letter':tok['letter'],
                         'musician':tok['musician'],
                         'inhospitality':tok['inhospitality'],
                         'distortion':tok['distortion'],
                         'thought':tok['thought'],
                         'descent':tok['descent'],
                         'hopelessness':tok['hopelessness'],
                         'acquittal':tok['acquittal'],
                         'lawyer':tok['lawyer'],
                         'minerals0metals':tok['minerals0metals'],
                         'pride':tok['pride'],
                         'disintegration':tok['disintegration'],
                         'track0and0field':tok['track0and0field'],
                         'depression':tok['depression'],
                         'violence':tok['violence'],
                         'instruments0of0punishment':tok['instruments0of0punishment'],
                         'willingness':tok['willingness'],
                         'worship':tok['worship'],
                         'photography':tok['photography'],
                         'tennis':tok['tennis'],
                         'sorcery':tok['sorcery'],
                         'imperfection':tok['imperfection'],
                         'unintelligence':tok['unintelligence'],
                         'odorlessness':tok['odorlessness'],
                         'extraction':tok['extraction'],
                         'trisection':tok['trisection'],
                         'warning':tok['warning'],
                         'shortcoming':tok['shortcoming'],
                         'theft':tok['theft'],
                         'circumscription':tok['circumscription'],
                         'mental0attitude':tok['mental0attitude'],
                         'verticalness':tok['verticalness'],
                         'unintelligibility':tok['unintelligibility'],
                         'ship0boat':tok['ship0boat'],
                         'relief':tok['relief'],
                         'closure':tok['closure'],
                         'substantiality':tok['substantiality'],
                         'colorlessness':tok['colorlessness'],
                         'facility':tok['facility'],
                         'narrow0mindedness':tok['narrow0mindedness'],
                         'weakness':tok['weakness'],
                         'sea0ocean':tok['sea0ocean'],
                         'cooking':tok['cooking'],
                         'servility':tok['servility'],
                         'talkativeness':tok['talkativeness'],
                         'subsequence':tok['subsequence'],
                         'refrigeration':tok['refrigeration'],
                         'relapse':tok['relapse'],
                         'gambling':tok['gambling'],
                         'satiety':tok['satiety'],
                         'debt':tok['debt'],
                         'overestimation':tok['overestimation'],
                         'thrift':tok['thrift'],
                         'deviation':tok['deviation'],
                         'school':tok['school'],
                         'uselessness':tok['uselessness'],
                         'opaqueness':tok['opaqueness'],
                         'agitation':tok['agitation'],
                         'region':tok['region'],
                         'dryness':tok['dryness'],
                         'cohesion':tok['cohesion'],
                         'frequency':tok['frequency'],
                         'blueness':tok['blueness'],
                         'the0clergy':tok['the0clergy'],
                         'intellect':tok['intellect'],
                         'fasting':tok['fasting'],
                         'nuclear0physics':tok['nuclear0physics'],
                         'death':tok['death'],
                         'incredulity':tok['incredulity'],
                         'exaggeration':tok['exaggeration'],
                         'light':tok['light'],
                         'noncohesion':tok['noncohesion'],
                         'horse0racing':tok['horse0racing'],
                         'imperfect0speech':tok['imperfect0speech'],
                         'instantaneousness':tok['instantaneousness'],
                         'solution':tok['solution'],
                         'decrease':tok['decrease'],
                         'exteriority':tok['exteriority'],
                         'spectator':tok['spectator'],
                         'excess':tok['excess'],
                         'hell':tok['hell'],
                         'prophets0religious0founders':tok['prophets0religious0founders'],
                         'composition':tok['composition'],
                         'participation':tok['participation'],
                         'humorousness':tok['humorousness'],
                         'town0city':tok['town0city'],
                         'modesty':tok['modesty'],
                         'politician':tok['politician'],
                         'room':tok['room'],
                         'oscillation':tok['oscillation'],
                         'inclusion':tok['inclusion'],
                         'clothing':tok['clothing'],
                         'inexcitability':tok['inexcitability'],
                         'boxing':tok['boxing'],
                         'evil0spirits':tok['evil0spirits'],
                         'library':tok['library'],
                         'smoothness':tok['smoothness'],
                         'liability':tok['liability'],
                         'intrinsicality':tok['intrinsicality'],
                         'plurality':tok['plurality'],
                         'asceticism':tok['asceticism'],
                         'overrunning':tok['overrunning'],
                         'approach':tok['approach'],
                         'orthodoxy':tok['orthodoxy'],
                         'merchandise':tok['merchandise'],
                         'arena':tok['arena'],
                         'poetry':tok['poetry'],
                         'love':tok['love'],
                         'food':tok['food'],
                         'unskillfulness':tok['unskillfulness'],
                         'confinement':tok['confinement'],
                         'yellowness':tok['yellowness'],
                         'accusation':tok['accusation'],
                         'moderation':tok['moderation'],
                         'humankind':tok['humankind'],
                         'healthfulness':tok['healthfulness'],
                         'dislike':tok['dislike'],
                         'whole':tok['whole'],
                         'shortness':tok['shortness'],
                         'eloquence':tok['eloquence'],
                         'electricity0magnetism':tok['electricity0magnetism'],
                         'patience':tok['patience'],
                         'taking':tok['taking'],
                         'bottom':tok['bottom'],
                         'dueness':tok['dueness'],
                         'aggravation':tok['aggravation'],
                         'prohibition':tok['prohibition'],
                         'unnervousness':tok['unnervousness'],
                         'spell0charm':tok['spell0charm'],
                         'guilt':tok['guilt'],
                         'condolence':tok['condolence'],
                         'retention':tok['retention'],
                         'nearness':tok['nearness'],
                         'teacher':tok['teacher'],
                         'disarrangement':tok['disarrangement'],
                         'expensiveness':tok['expensiveness'],
                         'stream':tok['stream'],
                         'disclosure':tok['disclosure'],
                         'inorganic0matter':tok['inorganic0matter'],
                         'intellectual':tok['intellectual'],
                         'writing':tok['writing'],
                         'phrase':tok['phrase'],
                         'duplication':tok['duplication'],
                         'physics':tok['physics'],
                         'the0environment':tok['the0environment'],
                         'vanity':tok['vanity'],
                         'fashion':tok['fashion'],
                         'rock':tok['rock'],
                         'uniformity':tok['uniformity'],
                         'forgiveness':tok['forgiveness'],
                         'angel0saint':tok['angel0saint'],
                         'theory0supposition':tok['theory0supposition'],
                         'arms':tok['arms'],
                         'affirmation':tok['affirmation'],
                         'poverty':tok['poverty'],
                         'obliquity':tok['obliquity'],
                         'freedom':tok['freedom'],
                         'lack0of0feeling':tok['lack0of0feeling'],
                         'punishment':tok['punishment'],
                         'five0and0over':tok['five0and0over'],
                         'cunning':tok['cunning'],
                         'continuity':tok['continuity'],
                         'simplicity':tok['simplicity'],
                         'unimportance':tok['unimportance'],
                         'temperance':tok['temperance'],
                         'warfare':tok['warfare'],
                         'combatant':tok['combatant'],
                         'evildoer':tok['evildoer'],
                         'comfort':tok['comfort'],
                         'foolishness':tok['foolishness'],
                         'disrespect':tok['disrespect'],
                         'front':tok['front'],
                         'absence':tok['absence'],
                         'sanctimony':tok['sanctimony'],
                         'dissimilarity':tok['dissimilarity'],
                         'repute':tok['repute'],
                         'misuse':tok['misuse'],
                         'manner0means':tok['manner0means'],
                         'endeavor':tok['endeavor'],
                         'pleasantness':tok['pleasantness'],
                         'silence':tok['silence'],
                         'grandiloquence':tok['grandiloquence'],
                         'conversion':tok['conversion'],
                         'unclothing':tok['unclothing'],
                         'completeness':tok['completeness'],
                         'travel':tok['travel'],
                         'jealousy':tok['jealousy'],
                         'right0side':tok['right0side'],
                         'predetermination':tok['predetermination'],
                         'communications':tok['communications'],
                         'animal0husbandry':tok['animal0husbandry'],
                         'straightness':tok['straightness'],
                         'imitation':tok['imitation'],
                         'threat':tok['threat'],
                         'narrowness0thinness':tok['narrowness0thinness'],
                         'ugliness':tok['ugliness'],
                         'indifference':tok['indifference'],
                         'side':tok['side'],
                         'mediation':tok['mediation'],
                         'perseverance':tok['perseverance'],
                         'unpleasure':tok['unpleasure'],
                         'ceramics':tok['ceramics'],
                         'belief':tok['belief'],
                         'regularity0of0recurrence':tok['regularity0of0recurrence'],
                         'the0universe0astronomy':tok['the0universe0astronomy'],
                         'divergence':tok['divergence'],
                         'arrogance':tok['arrogance'],
                         'aircraft':tok['aircraft'],
                         'social0convention':tok['social0convention'],
                         'explosive0noise':tok['explosive0noise'],
                         'curvature':tok['curvature'],
                         'graphic0arts':tok['graphic0arts'],
                         'habitation':tok['habitation'],
                         'aid':tok['aid'],
                         'ethics':tok['ethics'],
                         'flattery':tok['flattery'],
                         'cheerfulness':tok['cheerfulness'],
                         'powderiness0crumbliness':tok['powderiness0crumbliness'],
                         'sanctity':tok['sanctity'],
                         'indecency':tok['indecency'],
                         'irregularity0of0recurrence':tok['irregularity0of0recurrence'],
                         'submission':tok['submission'],
                         'fewness':tok['fewness'],
                         'changing0of0mind':tok['changing0of0mind'],
                         'unbelief':tok['unbelief'],
                         'accomplishment':tok['accomplishment'],
                         'adult0or0old0person':tok['adult0or0old0person'],
                         'revenge':tok['revenge'],
                         'forgetfulness':tok['forgetfulness'],
                         'record':tok['record'],
                         'event':tok['event'],
                         'speech':tok['speech'],
                         'shallowness':tok['shallowness'],
                         'council':tok['council'],
                         'circuitousness':tok['circuitousness'],
                         'anonymity':tok['anonymity'],
                         'plain0speech':tok['plain0speech'],
                         'religious0rites':tok['religious0rites'],
                         'plainness':tok['plainness'],
                         'anachronism':tok['anachronism'],
                         'mean':tok['mean'],
                         'absence0of0influence':tok['absence0of0influence'],
                         'divorce0widowhood':tok['divorce0widowhood'],
                         'beauty':tok['beauty'],
                         'reaction':tok['reaction'],
                         'dissent':tok['dissent'],
                         'unchastity':tok['unchastity'],
                         'football':tok['football'],
                         'greatness':tok['greatness'],
                         'effect':tok['effect'],
                         'social0class0and0status':tok['social0class0and0status'],
                         'ancestry':tok['ancestry'],
                         'humility':tok['humility'],
                         'exertion':tok['exertion'],
                         'body0of0land':tok['body0of0land'],
                         'repetition':tok['repetition'],
                         'tendency':tok['tendency'],
                         'cry0call':tok['cry0call'],
                         'disappointment':tok['disappointment'],
                         'latent0meaningfulness':tok['latent0meaningfulness'],
                         'lightness':tok['lightness'],
                         'attention':tok['attention'],
                         'pacification':tok['pacification'],
                         'possessor':tok['possessor'],
                         'sports':tok['sports'],
                         'breadth0thickness':tok['breadth0thickness'],
                         'disuse':tok['disuse'],
                         'promise':tok['promise'],
                         'wind':tok['wind'],
                         'extrinsicality':tok['extrinsicality'],
                         'memory':tok['memory'],
                         'existence':tok['existence'],
                         'misjudgment':tok['misjudgment'],
                         'inaction':tok['inaction'],
                         'quantity':tok['quantity'],
                         'sensations0of0touch':tok['sensations0of0touch'],
                         'uncommunicativeness':tok['uncommunicativeness'],
                         'insanity0mania':tok['insanity0mania'],
                         'three':tok['three'],
                         'recorder':tok['recorder'],
                         'discount':tok['discount'],
                         'part':tok['part'],
                         'prejudgment':tok['prejudgment'],
                         'reproduction0procreation':tok['reproduction0procreation'],
                         'order':tok['order'],
                         'earth0science':tok['earth0science'],
                         'gluttony':tok['gluttony'],
                         'strength':tok['strength'],
                         'elevation':tok['elevation'],
                         'opening':tok['opening'],
                         'dissuasion':tok['dissuasion'],
                         'consent':tok['consent'],
                         'darkness0dimness':tok['darkness0dimness'],
                         'impulse0impact':tok['impulse0impact'],
                         'specialty':tok['specialty'],
                         'victory':tok['victory'],
                         'resins0gums':tok['resins0gums'],
                         'banter':tok['banter'],
                         'nonobservance':tok['nonobservance'],
                         'multiformity':tok['multiformity'],
                         'shaft':tok['shaft'],
                         'inversion':tok['inversion'],
                         'philosophy':tok['philosophy'],
                         'strictness':tok['strictness'],
                         'obstinacy':tok['obstinacy'],
                         'pendency':tok['pendency'],
                         'benefactor':tok['benefactor'],
                         'request':tok['request'],
                         'fiction':tok['fiction'],
                         'penalty':tok['penalty'],
                         'superiority':tok['superiority'],
                         'absence0of0thought':tok['absence0of0thought'],
                         'reception':tok['reception'],
                         'escape':tok['escape'],
                         'materials':tok['materials'],
                         'lawlessness':tok['lawlessness'],
                         'receiving':tok['receiving'],
                         'envy':tok['envy'],
                         'impotence':tok['impotence'],
                         'cowardice':tok['cowardice'],
                         'purchase':tok['purchase'],
                         'combination':tok['combination'],
                         'infrequency':tok['infrequency'],
                         'disappearance':tok['disappearance'],
                         'resolution':tok['resolution'],
                         'distance0remoteness':tok['distance0remoteness'],
                         'nomenclature':tok['nomenclature'],
                         'authority':tok['authority'],
                         'aviator':tok['aviator'],
                         'length':tok['length'],
                         'teaching':tok['teaching'],
                         'particularity':tok['particularity'],
                         'birth':tok['birth'],
                         'respect':tok['respect'],
                         'payment':tok['payment'],
                         'depth':tok['depth'],
                         'sound':tok['sound'],
                         'unkindness0malevolence':tok['unkindness0malevolence'],
                         'pain':tok['pain'],
                         'diction':tok['diction'],
                         'precedence':tok['precedence'],
                         'liquefaction':tok['liquefaction'],
                         'alarm':tok['alarm'],
                         'fool':tok['fool'],
                         'health0care':tok['health0care'],
                         'messenger':tok['messenger'],
                         'perfection':tok['perfection'],
                         'aviation':tok['aviation'],
                         'hearing':tok['hearing'],
                         'nutrition':tok['nutrition'],
                         'friction':tok['friction'],
                         'heaven':tok['heaven'],
                         'variegation':tok['variegation'],
                         'language':tok['language'],
                         'color':tok['color'],
                         'oils0lubricants':tok['oils0lubricants'],
                         'space':tok['space'],
                         'bribery':tok['bribery'],
                         'motion':tok['motion'],
                         'timelessness':tok['timelessness'],
                         'disrepute':tok['disrepute'],
                         'production':tok['production'],
                         'pulling':tok['pulling'],
                         'apportionment':tok['apportionment'],
                         'oldness':tok['oldness'],
                         'relationship0by0marriage':tok['relationship0by0marriage'],
                         'bubble':tok['bubble'],
                         'vision':tok['vision'],
                         'newness':tok['newness'],
                         'information':tok['information'],
                         'jurisdiction':tok['jurisdiction'],
                         'container':tok['container'],
                         'contention':tok['contention'],
                         'nonpayment':tok['nonpayment'],
                         'interposition':tok['interposition'],
                         'heating':tok['heating'],
                         'arrangement':tok['arrangement'],
                         'courtesy':tok['courtesy'],
                         'goodness':tok['goodness'],
                         'defiance':tok['defiance'],
                         'entertainer':tok['entertainer'],
                         'repeal':tok['repeal'],
                         'swiftness':tok['swiftness'],
                         'refuge':tok['refuge'],
                         'receipts':tok['receipts'],
                         'foresight':tok['foresight'],
                         'mixture':tok['mixture'],
                         'discord':tok['discord'],
                         'bounds':tok['bounds'],
                         'caution':tok['caution'],
                         'inequality':tok['inequality'],
                         'crossing':tok['crossing'],
                         'relinquishment':tok['relinquishment'],
                         'allurement':tok['allurement'],
                         'direction0management':tok['direction0management'],
                         'hope':tok['hope'],
                         'opponent':tok['opponent'],
                         'commerce0economics':tok['commerce0economics'],
                         'eating':tok['eating'],
                         'sobriety':tok['sobriety'],
                         'undueness':tok['undueness'],
                         'relationship0by0blood':tok['relationship0by0blood'],
                         'correlation':tok['correlation'],
                         'discourtesy':tok['discourtesy'],
                         'preservation':tok['preservation'],
                         'prearrangement':tok['prearrangement'],
                         'femininity':tok['femininity'],
                         'computer0science':tok['computer0science'],
                         'inhabitant0native':tok['inhabitant0native'],
                         'model':tok['model'],
                         'avoidance':tok['avoidance'],
                         'occupation':tok['occupation'],
                         'motion0pictures':tok['motion0pictures'],
                         'copy':tok['copy'],
                         'attribution':tok['attribution'],
                         'lovemaking0endearment':tok['lovemaking0endearment'],
                         'masculinity':tok['masculinity'],
                         'meaninglessness':tok['meaninglessness'],
                         'government':tok['government'],
                         'abandonment':tok['abandonment'],
                         'sale':tok['sale'],
                         'imminence':tok['imminence'],
                         'figure0of0speech':tok['figure0of0speech'],
                         'impossibility':tok['impossibility'],
                         'extraneousness':tok['extraneousness'],
                         'involvement':tok['involvement'],
                         'touch':tok['touch'],
                         'impairment':tok['impairment'],
                         'support':tok['support'],
                         'subtraction':tok['subtraction'],
                         'ungrammaticalness':tok['ungrammaticalness'],
                         'sibilation':tok['sibilation'],
                         'grammar':tok['grammar'],
                         'transferal0transportation':tok['transferal0transportation'],
                         'progression':tok['progression'],
                         'television':tok['television'],
                         'choice':tok['choice'],
                         'marsh':tok['marsh'],
                         'injustice':tok['injustice'],
                         'intelligence0wisdom':tok['intelligence0wisdom'],
                         'light0source':tok['light0source'],
                         'brittleness0fragility':tok['brittleness0fragility'],
                         'nonreligiousness':tok['nonreligiousness'],
                         'semitransparency':tok['semitransparency'],
                         'insensibility':tok['insensibility'],
                         'chance':tok['chance'],
                         'analysis':tok['analysis'],
                         'covering':tok['covering'],
                         'failure':tok['failure'],
                         'book':tok['book'],
                         'safety':tok['safety'],
                         'pungency':tok['pungency'],
                         'blindness':tok['blindness'],
                         'ejection':tok['ejection'],
                         'compulsion':tok['compulsion'],
                         'wise0person':tok['wise0person'],
                         'judge0jury':tok['judge0jury'],
                         'furniture':tok['furniture'],
                         'duty':tok['duty'],
                         'interchange':tok['interchange'],
                         'use':tok['use'],
                         'anxiety':tok['anxiety'],
                         'golf':tok['golf'],
                         'lateness':tok['lateness'],
                         'water0travel':tok['water0travel'],
                         'disaccord':tok['disaccord'],
                         'bowling':tok['bowling'],
                         'resistance':tok['resistance'],
                         'atonement':tok['atonement'],
                         'incompleteness':tok['incompleteness'],
                         'opposition':tok['opposition'],
                         'unimaginativeness':tok['unimaginativeness'],
                         'correspondence':tok['correspondence'],
                         'news':tok['news'],
                         'automobile0racing':tok['automobile0racing'],
                         'sweetness':tok['sweetness'],
                         'change':tok['change'],
                         'bluster':tok['bluster'],
                         'arrival':tok['arrival'],
                         'comparison':tok['comparison'],
                         'justification':tok['justification'],
                         'credulity':tok['credulity'],
                         'impiety':tok['impiety'],
                         'the0present':tok['the0present'],
                         'radiation0radioactivity':tok['radiation0radioactivity'],
                         'thief':tok['thief'],
                         'theology':tok['theology'],
                         'precept':tok['precept'],
                         'inlet0gulf':tok['inlet0gulf'],
                         'quadrisection':tok['quadrisection'],
                         'compromise':tok['compromise'],
                         'wrongdoing':tok['wrongdoing'],
                         'intoxication0alcoholic0drink':tok['intoxication0alcoholic0drink'],
                         'operation':tok['operation'],
                         'joining':tok['joining'],
                         'concealment':tok['concealment'],
                         'reasoning':tok['reasoning'],
                         'assemblage':tok['assemblage'],
                         'destruction':tok['destruction'],
                         'insipidness':tok['insipidness'],
                         'highlands':tok['highlands'],
                         'faintness0of0sound':tok['faintness0of0sound'],
                         'season':tok['season'],
                         'prerogative':tok['prerogative'],
                         'disease':tok['disease'],
                         'liberality':tok['liberality'],
                         'unorthodoxy':tok['unorthodoxy'],
                         'distraction0confusion':tok['distraction0confusion'],
                         'remedy':tok['remedy'],
                         'plan':tok['plan'],
                         'hindrance':tok['hindrance'],
                         'inquiry':tok['inquiry'],
                         'servant0employee':tok['servant0employee'],
                         'history':tok['history'],
                         'nervousness':tok['nervousness'],
                         'visual0arts':tok['visual0arts'],
                         'layer':tok['layer'],
                         'intelligibility':tok['intelligibility'],
                         'deceiver':tok['deceiver'],
                         'state':tok['state'],
                         'bluntness':tok['bluntness'],
                         'imagination':tok['imagination'],
                         'size0largeness':tok['size0largeness'],
                         'selfishness':tok['selfishness'],
                         'taste0tastefulness':tok['taste0tastefulness'],
                         'therapy0medical0treatment':tok['therapy0medical0treatment'],
                         'repeated0sounds':tok['repeated0sounds'],
                         'amusement':tok['amusement'],
                         'workplace':tok['workplace'],"""
                         'groupn01':tok['groupn01'],
                         'thingn12':tok['thingn12'],
                         'measuren02':tok['measuren02'],
                         'changen06':tok['changen06'],
                         'objectn01':tok['objectn01'],
                         'substancen04':tok['substancen04'],
                         'causalagentn01':tok['causalagentn01'],
                         'relationn01':tok['relationn01'],
                         'mattern03':tok['mattern03'],
                         'horrorn02':tok['horrorn02'],
                         'communicationn02':tok['communicationn02'],
                         'psychologicalfeaturen01':tok['psychologicalfeaturen01'],
                         'setn02':tok['setn02'],
                         'processn06':tok['processn06'],
                         'attributen02':tok['attributen02'],
                         'verbweather':tok['verbweather'],
                         'nounsubstance':tok['nounsubstance'],
                         'nounprocess':tok['nounprocess'],
                         'nounTops':tok['nounTops'],
                         'verbmotion':tok['verbmotion'],
                         'nounfeeling':tok['nounfeeling'],
                         'nounstate':tok['nounstate'],
                         'verbstative':tok['verbstative'],
                         'verbbody':tok['verbbody'],
                         'nounlocation':tok['nounlocation'],
                         'nounshape':tok['nounshape'],
                         'verbchange':tok['verbchange'],
                         'nounevent':tok['nounevent'],
                         'verbcompetition':tok['verbcompetition'],
                         'nounfood':tok['nounfood'],
                         'verbemotion':tok['verbemotion'],
                         'nountime':tok['nountime'],
                         'verbsocial':tok['verbsocial'],
                         'nounbody':tok['nounbody'],
                         'nouncognition':tok['nouncognition'],
                         'noungroup':tok['noungroup'],
                         'nounact':tok['nounact'],
                         'advall':tok['advall'],
                         'nounquantity':tok['nounquantity'],
                         'nounartifact':tok['nounartifact'],
                         'verbconsumption':tok['verbconsumption'],
                         'verbpossession':tok['verbpossession'],
                         'verbperception':tok['verbperception'],
                         'adjall':tok['adjall'],
                         'nounplant':tok['nounplant'],
                         'verbcognition':tok['verbcognition'],
                         'nounperson':tok['nounperson'],
                         'adjpert':tok['adjpert'],
                         'nounattribute':tok['nounattribute'],
                         'nounanimal':tok['nounanimal'],
                         'verbcommunication':tok['verbcommunication'],
                         'nouncommunication':tok['nouncommunication'],
                         'nounmotive':tok['nounmotive'],
                         'adjppl':tok['adjppl'],
                         'nounpossession':tok['nounpossession'],
                         'nounobject':tok['nounobject'],
                         'verbcontact':tok['verbcontact'],
                         'nounrelation':tok['nounrelation'],
                         'verbcreation':tok['verbcreation'],
                         'nounphenomenon':tok['nounphenomenon'],
                         'rel':tok['rel'] if 'rel' in tok else EMPTY,
                         'sp1':sp1,
                         'slsb':slsb,
                         'srsb':srsb,
                         'sr2sb':sr2sb
                        }
                        for tok in GraphState.sent] # atomic features for current state

    def pcopy(self):
        return cPickle.loads(cPickle.dumps(self,-1))

    def is_terminal(self):
        """done traverse the graph"""
        return self.idx == -1

    def is_permissible(self,action):
        #TODO
        return True

    def is_possible_align(self,currentIdx,goldIdx,ref_graph):
        '''
        tmp_state = self.pcopy()
        oracle = __import__("oracle").DetOracleSC()
        next_action,label = oracle.give_ref_action(tmp_state,ref_graph)
        while tmp_state.beta:
            next_action['edge_label'] = label
            tmp_state = tmp_state.apply(next_action)
            next_action,label = oracle.give_ref_action(tmp_state,ref_graph)
        '''
        #ref_children = [ref_graph.abt_node_table[c] if c in ref_graph.abt_node_table else c for c in ref_graph.nodes[goldIdx].children]
        #return len(set(self.A.nodes[currentIdx].children) & set(ref_children)) > 1 or self.A.nodes[currentIdx].words[0][0].lower() == goldIdx
        if self.A.nodes[currentIdx].words[0].lower() in prep_list:
            return False
        return True

    def get_current_argset(self):
        if self.idx == START_ID:
            return set([])
        currentIdx = self.idx
        currentNode = self.get_current_node()
        currentGraph = self.A
        # record the core arguments current node(predicate) have
        return set(currentGraph.get_edge_label(currentIdx,c) for c in currentNode.children if currentGraph.get_edge_label(currentIdx,c).startswith('ARG'))
    def get_possible_actions(self,train):

        if self.idx == START_ID:
            return [{'type':NEXT2}]

        actions = []
        currentIdx = self.idx
        currentChildIdx = self.cidx
        currentNode = self.get_current_node()
        currentChild = self.get_current_child()
        currentGraph = self.A
        token_label_set = GraphState.model.token_label_set
        token_to_concept_table = GraphState.model.token_to_concept_table
        tag_codebook = GraphState.model.tag_codebook

        if isinstance(currentIdx,int):
            current_tok_lemma = ','.join(tok['lemma'] for tok in GraphState.sent if tok['id'] in range(currentNode.start,currentNode.end))
            current_tok_form = ','.join(tok['form'] for tok in GraphState.sent if tok['id'] in range(currentNode.start,currentNode.end))
            current_tok_ne = GraphState.sent[currentIdx]['ne']
        else:
            current_tok_form = ABT_TOKEN['form']
            current_tok_lemma = ABT_TOKEN['lemma'] #if currentIdx != START_ID else START_TOKEN['lemma']
            current_tok_ne = ABT_TOKEN['ne'] #if currentIdx != START_ID else START_TOKEN['ne']

        #if self.action_history and self.action_history[-1]['type'] in [REPLACEHEAD,NEXT2,DELETENODE] and currentNode.num_parent_infer_in_chain < 3 and currentNode.num_parent_infer == 0:
            #actions.extend([{'type':INFER,'tag':z} for z in tag_codebook['ABTTag'].labels()])

        if currentChildIdx: # beta not empty
            #all_candidate_edge_labels = GraphState.model.rel_codebook.labels()

            #if current_tok_lemma in token_label_set:
            #    all_candidate_edge_labels.extend(list(token_label_set[current_tok_lemma]))
            #elif current_tok_ne not in ['O','NUMBER']:
            #    all_candidate_edge_labels.extend(list(token_label_set[current_tok_ne]))
                #all_candidate_tags.extend(GraphState.model.tag_codebook['ETag'].labels())
            #else:
            #    all_candidate_tags.append(current_tok_lemma)  # for decoding

            if currentChildIdx == START_ID:
                if currentNode.num_parent_infer_in_chain < 3 and currentNode.num_parent_infer == 0:
                    actions = [{'type':NEXT1},{'type':INFER}]
                else:
                    actions = [{'type':NEXT1}]
                return actions

            if currentIdx != 0: # not root
                if not currentChild.SWAPPED:
                    #actions.extend([{'type':MERGE},{'type':REPLACEHEAD}])
                    ##actions.extend([{'type':NEXT1,'edge_label':y} for y in all_candidate_edge_labels])
                    #actions.append({'type':NEXT1})
                #else:
                    #actions.extend([{'type':MERGE},{'type':REPLACEHEAD},{'type':SWAP}])
                    #actions.append({'type':NEXT1})
                    ##if len(currentChild.parents) > 1:
                    ##actions.append({'type':REATTACH,'parent_to_attach':None}) # this equals delete edge
                    actions.append({'type':SWAP})
                    actions.extend([{'type':REATTACH,'parent_to_attach':p} for p in currentGraph.get_possible_parent_constrained(currentIdx,currentChildIdx)])

                    #actions.extend([{'type':NEXT1,'edge_label':y} for y in all_candidate_edge_labels])

                if isinstance(currentIdx,int) and isinstance(currentChildIdx,int):
                    actions.append({'type':MERGE})
                actions.extend([{'type':NEXT1},{'type':REPLACEHEAD}])
                actions.extend({'type':REENTRANCE,'parent_to_add':x} for x in currentGraph.get_possible_reentrance_constrained(currentIdx,currentChildIdx))
            else:
                actions.extend([{'type':NEXT1}])
                #if len(currentChild.parents) > 1:
                #actions.append({'type':REATTACH,'parent_to_attach':None}) # this equals delete edge
                actions.extend([{'type':REATTACH,'parent_to_attach':p} for p in currentGraph.get_possible_parent_constrained(currentIdx,currentChildIdx)])
                #actions.extend({'type':ADDCHILD,'child_type':x} for x in currentGraph.get_possible_children_unconstrained(currentIdx))
        else:
            all_candidate_tags = []
            # MOD
            if current_tok_lemma in token_to_concept_table:
                all_candidate_tags.extend(list(token_to_concept_table[current_tok_lemma]))
                #all_candidate_tags.append(current_tok_lemma.lower())
            elif isinstance(currentIdx,int) and (current_tok_ne not in ['O','NUMBER'] or currentNode.end - currentNode.start > 1):
                all_candidate_tags.extend(list(token_to_concept_table[current_tok_ne]))
                #all_candidate_tags.append(current_tok_lemma.lower())
                #all_candidate_tags.extend(GraphState.model.tag_codebook['ETag'].labels())
            elif current_tok_lemma == ABT_TOKEN['lemma']:
                #all_candidate_tags.extend(tag_codebook['ABTTag'].labels())
                pass
                #all_candidate_tags.extend(currentGraph.nodes[currentIdx].tag)
            else:
                all_candidate_tags.append(current_tok_lemma.lower())  # for decoding

            if isinstance(currentIdx,int) and 'frmset' in GraphState.sent[currentIdx] \
               and GraphState.sent[currentIdx]['frmset'] not in all_candidate_tags:
                all_candidate_tags.append(GraphState.sent[currentIdx]['frmset'])


            if not currentNode.children and currentIdx != 0:
                actions.append({'type':DELETENODE})
            actions.append({'type':NEXT2})
            actions.extend({'type':NEXT2,'tag':z} for z in all_candidate_tags)

        return actions

    def get_node_context(self,idx):
        # first parent of current node
        if self.A.nodes[idx].parents:
            p1 = GraphState.sent[self.A.nodes[idx].parents[0]] if isinstance(self.A.nodes[idx].parents[0],int) else ABT_TOKEN
            p1_brown_repr = BROWN_CLUSTER[p1['form']]
            p1['brown4'] = p1_brown_repr[:4] if len(p1_brown_repr) > 3 else p1_brown_repr
            p1['brown6'] = p1_brown_repr[:6] if len(p1_brown_repr) > 5 else p1_brown_repr
            p1['brown10'] = p1_brown_repr[:10] if len(p1_brown_repr) > 9 else p1_brown_repr
            p1['brown20'] = p1_brown_repr[:20] if len(p1_brown_repr) > 19 else p1_brown_repr
            ### i don't know why i need this?
            if 'nec' not in p1:
                p1['nec'] = "O";
            ########## I took the w2v P1 feature out so this should probably be fine...
            #if 'w2v' not in p1:
            #    p1['w2v'] = EMPTY;
        else:
            p1 = NOT_ASSIGNED
        if isinstance(idx,int):
            prs1 = GraphState.sent[idx-1] if idx > 0 else NOT_ASSIGNED
            prs2 = GraphState.sent[idx-2] if idx > 1 else NOT_ASSIGNED
        else:
            prs1 = ABT_TOKEN
            prs2 = ABT_TOKEN


        # immediate left sibling, immediate right sibling and second right sibling
        if p1 != NOT_ASSIGNED and len(self.A.nodes[self.A.nodes[idx].parents[0]].children) > 1:
            children = self.A.nodes[self.A.nodes[idx].parents[0]].children
            idx_order = sorted(children).index(idx)
            if idx_order > 0:
                lsb = GraphState.sent[children[idx_order-1]] if isinstance(children[idx_order-1],int) else ABT_TOKEN
            else:
                lsb = NOT_ASSIGNED
            if idx_order < len(children)-1:
                rsb = GraphState.sent[children[idx_order+1]] if isinstance(children[idx_order+1],int) else ABT_TOKEN
            else:
                rsb = NOT_ASSIGNED
            if idx_order < len(children)-2:
                r2sb = GraphState.sent[children[idx_order+2]] if isinstance(children[idx_order+2],int) else ABT_TOKEN
            else:
                r2sb = NOT_ASSIGNED
        else:
            lsb = EMPTY
            rsb = EMPTY
            r2sb = EMPTY

        return prs2,prs1,p1,lsb,rsb,r2sb

    def get_feature_context_window(self,action):
        """context window for current node and its child"""
        def isprep(token):
            return token['pos'] == 'IN' and token['rel'] == 'prep'
        def delta_func(tag_to_predict,tok_form):
            if isinstance(tag_to_predict,(ConstTag,ETag)):
                return 'ECTag'
            else:
                tok_form = tok_form.lower()
                tag_lemma = tag_to_predict.split('-')[0]
                if tag_lemma == tok_form:
                    return '1'
                i=0
                slength = len(tag_lemma) if len(tag_lemma) < len(tok_form) else len(tok_form)
                while i < slength and tag_lemma[i] == tok_form[i]:
                    i += 1
                if i == 0:
                    return '0'
                elif tok_form[i:]:
                    return tok_form[i:]
                elif tag_lemma[i:]:
                    return tag_lemma[i:]
                else:
                    assert False

        s0_atomics = GraphState.sent[self.idx].copy() if isinstance(self.idx,int) else ABT_TOKEN #GraphState.abt_tokens[self.idx]
        s0_brown_repr = BROWN_CLUSTER[s0_atomics['form']]
        s0_atomics['brown4'] = s0_brown_repr[:4] if len(s0_brown_repr) > 3 else s0_brown_repr
        s0_atomics['brown6'] = s0_brown_repr[:6] if len(s0_brown_repr) > 5 else s0_brown_repr
        s0_atomics['brown8'] = s0_brown_repr[:8] if len(s0_brown_repr) > 7 else s0_brown_repr
        s0_atomics['brown10'] = s0_brown_repr[:10] if len(s0_brown_repr) > 9 else s0_brown_repr
        s0_atomics['brown20'] = s0_brown_repr[:20] if len(s0_brown_repr) > 19 else s0_brown_repr


        #s0_atomics['pfx'] = s0_atomics['form'][:4] if len(s0_atomics['form']) > 3 else s0_atomics['form']
        sprs2,sprs1,sp1,slsb,srsb,sr2sb=self.get_node_context(self.idx)
        s0_atomics['prs1']=sprs1
        s0_atomics['prs2']=sprs2
        s0_atomics['p1']=sp1
        s0_atomics['lsb']=slsb
        s0_atomics['rsb']=srsb
        s0_atomics['r2sb']=sr2sb
        s0_atomics['len']=self.A.nodes[self.idx].end - self.A.nodes[self.idx].start if isinstance(self.idx,int) else NOT_ASSIGNED
        #s0_atomics['cap']=s0_atomics['form'].istitle()
        s0_atomics['dch']=sorted([GraphState.sent[j]['form'].lower() if isinstance(j,int) else ABT_FORM for j in self.A.nodes[self.idx].del_child])
        s0_atomics['reph']=sorted([GraphState.sent[j]['form'].lower() if isinstance(j,int) else ABT_FORM for j in self.A.nodes[self.idx].rep_parent])
        #s0_atomics['nech'] = len(set(GraphState.sent[j]['ne'] if isinstance(j,int) else ABT_NE for j in self.A.nodes[self.idx].children) & INFER_NETAG) > 0
        #s0_atomics['isnom'] = s0_atomics['lemma'] in NOMLIST

        core_args = set([self.A.get_edge_label(self.idx,child) for child in self.A.nodes[self.idx].children if self.A.get_edge_label(self.idx,child).startswith('ARG') and child != self.cidx])
        s0_atomics['lsl']=str(sorted(core_args)) # core argument
        s0_atomics['arg0']='ARG0' in core_args
        s0_atomics['arg1']='ARG1' in core_args
        s0_atomics['arg2']='ARG2' in core_args

        # prop feature
        s0_atomics['frmset']=GraphState.sent[self.idx]['frmset'] if isinstance(self.idx,int) and 'frmset' in GraphState.sent[self.idx] else NOT_ASSIGNED

        # mod here
        # next2 specific features
        if not self.cidx:
            if 'tag' in action: # next2
                tag_to_predict = action['tag']
                s0_atomics['eqfrmset'] = s0_atomics['frmset'] == tag_to_predict if s0_atomics['frmset'] is not NOT_ASSIGNED else NOT_ASSIGNED
                s0_atomics['txv'] = len(tag_to_predict.split('-'))==2
                s0_atomics['txn'] = isinstance(tag_to_predict,ETag)
                s0_atomics['txdelta'] = delta_func(tag_to_predict,s0_atomics['form'])
            else:
                s0_atomics['txv'] = NOT_ASSIGNED
                s0_atomics['txn'] = NOT_ASSIGNED
                s0_atomics['txdelta'] = NOT_ASSIGNED
                s0_atomics['eqfrmset'] = NOT_ASSIGNED
            s0_atomics['isleaf'] = len(self.A.nodes[self.idx].children) == 0
        else:
            s0_atomics['txv'] = NOT_APPLY
            s0_atomics['txn'] = NOT_APPLY
            s0_atomics['txdelta'] = NOT_APPLY
            s0_atomics['eqfrmset'] = NOT_APPLY
            s0_atomics['isleaf'] = NOT_APPLY

        s0_args = None
        s0_prds = None
        if isinstance(self.idx,int) and GraphState.sent[self.idx].get('args',{}):
            s0_args = GraphState.sent[self.idx]['args']
        if isinstance(self.idx,int) and GraphState.sent[self.idx].get('pred',{}):
            s0_prds = GraphState.sent[self.idx]['pred']

        if self.cidx and self.cidx != START_ID:
            b0_atomics = GraphState.sent[self.cidx].copy() if isinstance(self.cidx,int) else ABT_TOKEN #GraphState.abt_tokens[self.cidx]
            b0_brown_repr = BROWN_CLUSTER[b0_atomics['form']]
            b0_atomics['brown4'] = b0_brown_repr[:4] if len(b0_brown_repr) > 3 else b0_brown_repr
            b0_atomics['brown6'] = b0_brown_repr[:6] if len(b0_brown_repr) > 5 else b0_brown_repr
            b0_atomics['brown8'] = b0_brown_repr[:8] if len(b0_brown_repr) > 7 else b0_brown_repr
            b0_atomics['brown10'] = b0_brown_repr[:10] if len(b0_brown_repr) > 9 else b0_brown_repr
            b0_atomics['brown20'] = b0_brown_repr[:20] if len(b0_brown_repr) > 19 else b0_brown_repr
            b0_atomics['concept'] = self.A.nodes[self.cidx].tag
            bprs2,bprs1,bp1,blsb,brsb,br2sb = self.get_node_context(self.cidx)
            b0_atomics['prs1']=bprs1
            b0_atomics['prs2']=bprs2
            b0_atomics['p1']=bp1
            b0_atomics['lsb']=blsb
            b0_atomics['rsb']=brsb
            b0_atomics['r2sb']=br2sb
            b0_atomics['nswp']=self.A.nodes[self.cidx].num_swap
            b0_atomics['reph']=sorted([GraphState.sent[rp]['form'] if isinstance(rp,int) else ABT_FORM for rp in self.A.nodes[self.cidx].rep_parent])
            b0_atomics['len']=self.A.nodes[self.cidx].end - self.A.nodes[self.cidx].start if isinstance(self.cidx,int) else NOT_ASSIGNED
            b0_atomics['dch']=sorted([GraphState.sent[j]['form'].lower() if isinstance(j,int) else ABT_FORM for j in self.A.nodes[self.cidx].del_child])
            b0_atomics['eqne']=(s0_atomics['ne']==b0_atomics['ne'] and b0_atomics['ne'] in PRE_MERGE_NETAG)
            b0_atomics['isne']=b0_atomics['ne'] in PRE_MERGE_NETAG
            b0_atomics['hastrace'] = len(self.A.nodes[self.cidx].incoming_traces) > 0

            # prop feature
            b0_atomics['isarg']=self.cidx in s0_args if s0_args else NOT_ASSIGNED
            b0_atomics['arglabel']=s0_args[self.cidx] if b0_atomics['isarg'] else NOT_ASSIGNED

            b0_atomics['isprd']=self.cidx in s0_prds if s0_prds else NOT_ASSIGNED
            b0_atomics['prdlabel']=s0_prds[self.cidx] if b0_atomics['isprd'] else NOT_ASSIGNED

            if isinstance(self.cidx,int) and isinstance(self.idx,int):
                path,direction = GraphState.deptree.get_path(self.cidx,self.idx)
                if self.A.nodes[self.idx].end - self.A.nodes[self.idx].start > 1:
                    path_pos_str = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) for i in path[1:-1] if i not in range(self.A.nodes[self.idx].start,self.A.nodes[self.idx].end)]
                    path_x_str_pp = [('X','X') if not isprep(GraphState.sent[i]) else GraphState.sent[i]['form'] for i in path[1:-1] if i not in range(self.A.nodes[self.idx].start,self.A.nodes[self.idx].end)]
                else:
                    path_pos_str = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) for i in path[1:-1]]
                    path_x_str_pp = [('X','X') if not isprep(GraphState.sent[i]) else GraphState.sent[i]['form']  for i in path[1:-1]]
                path_pos_str.insert(0,GraphState.sent[path[0]]['rel'])
                path_pos_str.append(GraphState.sent[path[-1]]['rel'])

                path_x_str_pp.insert(0,GraphState.sent[path[0]]['rel'])
                path_x_str_pp.append(GraphState.sent[path[-1]]['rel'])

                b0_atomics['pathp'] = path_pos_str
                b0_atomics['pathprep'] = path_x_str_pp
                b0_atomics['pathpwd'] = str(path_pos_str) + direction
                b0_atomics['pathprepwd'] = str(path_x_str_pp) + direction
            else:
                b0_atomics['pathp'] = EMPTY
                b0_atomics['pathprep'] = EMPTY
                b0_atomics['pathpwd'] = EMPTY
                b0_atomics['pathprepwd'] = EMPTY

            b0_atomics['apathx'] = EMPTY
            b0_atomics['apathp'] = EMPTY
            b0_atomics['apathprep'] = EMPTY
            b0_atomics['apathxwd'] = EMPTY
            b0_atomics['apathpwd'] = EMPTY
            b0_atomics['apathprepwd'] = EMPTY
        else:
            b0_atomics = EMPTY


        if action['type'] in [REATTACH,REENTRANCE]:
            #child_to_add = action['child_to_add']
            if action['type'] == REATTACH:
                parent_to_attach = action['parent_to_attach']
            else:
                parent_to_attach = action['parent_to_add']
            if parent_to_attach is not None:
                a0_atomics = GraphState.sent[parent_to_attach].copy() if isinstance(parent_to_attach,int) else ABT_TOKEN #GraphState.abt_tokens[parent_to_attach]
                a0_brown_repr = BROWN_CLUSTER[a0_atomics['form']]
                a0_atomics['brown4'] = a0_brown_repr[:4] if len(a0_brown_repr) > 3 else a0_brown_repr
                a0_atomics['brown6'] = a0_brown_repr[:6] if len(a0_brown_repr) > 5 else a0_brown_repr
                a0_atomics['brown8'] = a0_brown_repr[:8] if len(a0_brown_repr) > 7 else a0_brown_repr
                a0_atomics['brown10'] = a0_brown_repr[:10] if len(a0_brown_repr) > 9 else a0_brown_repr
                a0_atomics['brown20'] = a0_brown_repr[:20] if len(a0_brown_repr) > 19 else a0_brown_repr
                a0_atomics['concept'] = self.A.nodes[parent_to_attach].tag
                aprs2,aprs1,ap1,alsb,arsb,ar2sb = self.get_node_context(parent_to_attach)

                a0_atomics['p1']=ap1
                a0_atomics['lsb']=alsb
                a0_atomics['rsb']=arsb
                a0_atomics['r2sb']=ar2sb
                a0_atomics['nswp']=self.A.nodes[parent_to_attach].num_swap
                a0_atomics['isne']=a0_atomics['ne'] is not 'O'


                itr = list(self.A.nodes[self.cidx].incoming_traces)
                tr = [t for r,t in itr]
                a0_atomics['istrace'] = parent_to_attach in tr if len(tr) > 0 else EMPTY
                a0_atomics['rtr'] = itr[tr.index(parent_to_attach)][0] if parent_to_attach in tr else EMPTY
                a0_atomics['hasnsubj'] = b0_atomics['rel'] in set(GraphState.sent[c]['rel'] for c in self.A.nodes[parent_to_attach].children if isinstance(c,int))
                #a0_atomics['iscycle'] = parent_to_attach in self.A.nodes[self.cidx].children or parent_to_attach in self.A.nodes[self.cidx].parents

                # prop feature
                b0_prds = None
                b0_args = None
                if isinstance(self.cidx,int) and GraphState.sent[self.cidx].get('pred',{}):
                    b0_prds = GraphState.sent[self.cidx]['pred']
                if isinstance(self.cidx,int) and GraphState.sent[self.cidx].get('args',{}):
                    b0_args = GraphState.sent[self.cidx]['args']

                a0_atomics['isprd']=parent_to_attach in b0_prds if b0_prds else NOT_ASSIGNED
                a0_atomics['prdlabel']=b0_prds[parent_to_attach] if a0_atomics['isprd'] else NOT_ASSIGNED

                a0_atomics['isarg']=parent_to_attach in b0_args if b0_args else NOT_ASSIGNED
                a0_atomics['arglabel']=b0_args[parent_to_attach] if a0_atomics['isarg'] else NOT_ASSIGNED

                if isinstance(self.cidx,int) and isinstance(parent_to_attach,int):
                    path,direction = GraphState.deptree.get_path(self.cidx,parent_to_attach)
                #path_x_str=[(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) for i in path[1:-1]]
                    if self.A.nodes[parent_to_attach].end - self.A.nodes[parent_to_attach].start > 1:
                        apath_x_str = [('X','X') for i in path[1:-1] if i not in range(self.A.nodes[parent_to_attach].start,self.A.nodes[parent_to_attach].end)]
                        apath_pos_str = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) for i in path[1:-1] if i not in range(self.A.nodes[parent_to_attach].start,self.A.nodes[parent_to_attach].end)]
                        apath_pos_str_pp = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) if not isprep(GraphState.sent[i]) else GraphState.sent[i]['form'] for i in path[1:-1] if i not in range(self.A.nodes[parent_to_attach].start,self.A.nodes[parent_to_attach].end)]
                    else:
                        apath_x_str = [('X','X') for i in path[1:-1]]
                        apath_pos_str = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) for i in path[1:-1]]
                        apath_pos_str_pp = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) if not isprep(GraphState.sent[i]) else GraphState.sent[i]['form'] for i in path[1:-1]]
                    apath_x_str.insert(0,GraphState.sent[path[0]]['rel'])
                    apath_x_str.append(GraphState.sent[path[-1]]['rel'])

                    apath_pos_str.insert(0,GraphState.sent[path[0]]['rel'])
                    apath_pos_str.append(GraphState.sent[path[-1]]['rel'])

                    apath_pos_str_pp.insert(0,GraphState.sent[path[0]]['rel'])
                    apath_pos_str_pp.append(GraphState.sent[path[-1]]['rel'])

            #path_label_str = [GraphState.sent[i]['rel'] for i in path] # dependency label
            #path_lemma_str.insert(0,GraphState.sent[path[0]]['rel'])
            #path_lemma_str.append(GraphState.sent[path[-1]]['rel'])
                    b0_atomics['apathx'] = apath_x_str
                    b0_atomics['apathp'] = apath_pos_str
                    b0_atomics['apathprep'] = apath_pos_str_pp
                    b0_atomics['apathxwd'] = str(apath_x_str) + direction
                    b0_atomics['apathpwd'] = str(apath_pos_str) + direction
                    b0_atomics['apathprepwd'] = str(apath_pos_str_pp) + direction
            #a0_atomics['pathl'] = path_label_str
                else:
                    b0_atomics['pathp'] = EMPTY
                    b0_atomics['pathprep'] = EMPTY
                    b0_atomics['pathpwd'] = EMPTY
                    b0_atomics['pathprepwd'] = EMPTY
                    b0_atomics['apathx'] = EMPTY
                    b0_atomics['apathp'] = EMPTY
                    b0_atomics['apathprep'] = EMPTY
                    b0_atomics['apathxwd'] = EMPTY
                    b0_atomics['apathpwd'] = EMPTY
                    b0_atomics['apathprepwd'] = EMPTY

            else:
                a0_atomics = EMPTY
        else:
            a0_atomics = EMPTY
            #a0_atomics = s0_atomics

        '''
        if action['type'] == REENTRANCE:
            parent_to_add = action['parent_to_add']
            itr = list(self.A.nodes[self.cidx].incoming_traces)
            tr = [t for r,t in itr]
            a0_atomics['istrace'] = parent_to_add in tr if len(tr) > 0 else EMPTY
            #a0_atomics['rtr'] = itr[tr.index(parent_to_add)][0] if parent_to_add in tr else EMPTY
        else:
            a0_atomics = EMPTY
        '''

        if self.cidx == START_ID:
            s0_atomics['nech'] = len(set(GraphState.sent[j]['ne'] if isinstance(j,int) else ABT_NE for j in self.A.nodes[self.idx].children) & INFER_NETAG) > 0
            s0_atomics['isnom'] = s0_atomics['lemma'].lower() in NOMLIST
            s0_atomics['concept']=self.A.nodes[self.idx].tag
            if self.A.nodes[self.idx].children:
                c1 = self.A.nodes[self.idx].children[0]
                s0_atomics['c1lemma'] = GraphState.sent[c1]['lemma'].lower() if isinstance(c1,int) else ABT_LEMMA
                s0_atomics['c1dl'] = GraphState.sent[c1]['rel'] if isinstance(c1,int) else ABT_LEMMA
            else:
                s0_atomics['c1lemma'] = EMPTY
                s0_atomics['c1dl'] = EMPTY
        else:
            s0_atomics['c1lemma'] = NOT_APPLY#EMPTY
            s0_atomics['concept'] = NOT_APPLY#EMPTY
            s0_atomics['nech'] = NOT_APPLY#EMPTY
            s0_atomics['isnom'] = NOT_APPLY#EMPTY
            s0_atomics['c1dl'] = NOT_APPLY#EMPTY

        '''
        if action['type'] == REENTRANCE and 'parent_to_add' in action: # reattach
            #child_to_add = action['child_to_add']
            parent_to_add = action['parent_to_add']
            r0_atomics = GraphState.sent[parent_to_add]
            rprs2,rprs1,rp1,rlsb,rrsb,rr2sb = self.get_node_context(parent_to_add)
            r0_atomics['p1']=rp1
            r0_atomics['lsb']=rlsb
            r0_atomics['rsb']=rrsb
            r0_atomics['r2sb']=rr2sb
            r0_atomics['nswp']=self.A.nodes[parent_to_add].num_swap
            r0_atomics['isne']=r0_atomics['ne'] is not 'O'
            #path,direction = self.A.get_path(self.cidx,parent_to_attach)
            path,direction = GraphState.deptree.get_path(self.cidx,parent_to_attach)
            #path_x_str=[(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) for i in path[1:-1]]
            if self.A.nodes[parent_to_attach].end - self.A.nodes[parent_to_attach].start > 1:
                apath_x_str = [('X','X') for i in path[1:-1] if i not in range(self.A.nodes[parent_to_attach].start,self.A.nodes[parent_to_attach].end)]
                apath_pos_str = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) for i in path[1:-1] if i not in range(self.A.nodes[parent_to_attach].start,self.A.nodes[parent_to_attach].end)]
                apath_pos_str_pp = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) if not isprep(GraphState.sent[i]) else GraphState.sent[i]['form'] for i in path[1:-1] if i not in range(self.A.nodes[parent_to_attach].start,self.A.nodes[parent_to_attach].end)]
            else:
                apath_x_str = [('X','X') for i in path[1:-1]]
                apath_pos_str = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) for i in path[1:-1]]
                apath_pos_str_pp = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) if not isprep(GraphState.sent[i]) else GraphState.sent[i]['form'] for i in path[1:-1]]
            apath_x_str.insert(0,GraphState.sent[path[0]]['rel'])
            apath_x_str.append(GraphState.sent[path[-1]]['rel'])

            apath_pos_str.insert(0,GraphState.sent[path[0]]['rel'])
            apath_pos_str.append(GraphState.sent[path[-1]]['rel'])

            apath_pos_str_pp.insert(0,GraphState.sent[path[0]]['rel'])
            apath_pos_str_pp.append(GraphState.sent[path[-1]]['rel'])

            #path_label_str = [GraphState.sent[i]['rel'] for i in path] # dependency label
            #path_lemma_str.insert(0,GraphState.sent[path[0]]['rel'])
            #path_lemma_str.append(GraphState.sent[path[-1]]['rel'])
            b0_atomics['apathx'] = apath_x_str
            b0_atomics['apathp'] = apath_pos_str
            b0_atomics['apathprep'] = apath_pos_str_pp
            b0_atomics['apathxwd'] = str(apath_x_str) + direction
            b0_atomics['apathpwd'] = str(apath_pos_str) + direction
            b0_atomics['apathprepwd'] = str(apath_pos_str_pp) + direction
            #a0_atomics['pathl'] = path_label_str
        else:
            a0_atomics = EMPTY
        '''

        return (s0_atomics,b0_atomics,a0_atomics)

    def get_gold_edge_graph(self):
        gold_edge_graph = copy.deepcopy(self.A)
        parsed_tuples = gold_edge_graph.tuples()
        gold_tuples = self.gold_graph.tuples()

        for t_tuple in parsed_tuples:
            if t_tuple in gold_tuples:
                gold_arc_label = self.gold_graph.get_edge_label(t_tuple[0],t_tuple[1])
                gold_edge_graph.set_edge_label(t_tuple[0],t_tuple[1],gold_arc_label)

        return gold_edge_graph

    def get_gold_tag_graph(self):
        gold_tag_graph = copy.deepcopy(self.A)
        for nid in gold_tag_graph.nodes.keys()[:]:
            if nid in self.gold_graph.nodes:
                gold_tag_label = self.gold_graph.get_node_tag(nid)
                gold_tag_graph.set_node_tag(nid,gold_tag_label)
        return gold_tag_graph

    def get_gold_label_graph(self):
        gold_label_graph = copy.deepcopy(self.A)
        parsed_tuples = gold_label_graph.tuples()
        gold_tuples = self.gold_graph.tuples()
        for t_tuple in parsed_tuples:
            if t_tuple in gold_tuples:
                gold_arc_label = self.gold_graph.get_edge_label(t_tuple[0],t_tuple[1])
                gold_label_graph.set_edge_label(t_tuple[0],t_tuple[1],gold_arc_label)
                gold_tag_label1 = self.gold_graph.get_node_tag(t_tuple[0])
                gold_label_graph.set_node_tag(t_tuple[0],gold_tag_label1)
                gold_tag_label2 = self.gold_graph.get_node_tag(t_tuple[1])
                gold_label_graph.set_node_tag(t_tuple[1],gold_tag_label2)
        return gold_label_graph

    def evaluate(self):
        num_correct_arcs = .0
        num_correct_labeled_arcs = .0

        parsed_tuples = self.A.tuples()
        if self.verbose > 1:
            print >> sys.stderr, 'Parsed tuples:'+str(parsed_tuples)
        num_parsed_arcs = len(parsed_tuples)
        gold_tuples = self.gold_graph.tuples()
        num_gold_arcs = len(gold_tuples)

        num_correct_tags = .0
        num_parsed_tags = .0
        num_gold_tags = .0
        visited_nodes = set()
        for t_tuple in parsed_tuples:
            p,c = t_tuple
            p_p,c_p = p,c
            if p in self.A.abt_node_table: p = self.A.abt_node_table[p]
            if c in self.A.abt_node_table: c = self.A.abt_node_table[c]
            if p_p not in visited_nodes:
                visited_nodes.add(p_p)
                p_tag = self.A.get_node_tag(p_p)
                if p in self.gold_graph.nodes:
                    g_p_tag = self.gold_graph.get_node_tag(p)
                    if p_tag == g_p_tag:# and not (isinstance(g_p_tag,(ETag,ConstTag)) or re.match('\w+-\d+',g_p_tag)): #and isinstance(g_p_tag,(ETag,ConstTag)):
                        num_correct_tags += 1.0
                    else:
                        self.A.nodes_error_table[p_p]=NODE_TYPE_ERROR
                else:
                    self.A.nodes_error_table[p_p]=NODE_MATCH_ERROR
                #    if p_tag == NULL_TAG:
                #        num_correct_tags += 1.0
            if c_p not in visited_nodes:
                visited_nodes.add(c_p)
                c_tag = self.A.get_node_tag(c_p)
                if c in self.gold_graph.nodes:
                    g_c_tag = self.gold_graph.get_node_tag(c)
                    if c_tag == g_c_tag:# and not (isinstance(g_c_tag,(ETag,ConstTag)) or re.match('\w+-\d+',g_c_tag)): #and isinstance(g_c_tag,(ETag,ConstTag)):
                        num_correct_tags += 1.0
                    else:
                        self.A.nodes_error_table[c_p]=NODE_TYPE_ERROR
                else:
                    self.A.nodes_error_table[c_p]=NODE_MATCH_ERROR
                #else:
                #    if c_tag == NULL_TAG:
                #        num_correct_tags += 1.0

            if (p,c) in gold_tuples:
                num_correct_arcs += 1.0
                parsed_arc_label = self.A.get_edge_label(p_p,c_p)
                gold_arc_label = self.gold_graph.get_edge_label(p,c)
                if parsed_arc_label == gold_arc_label:
                    num_correct_labeled_arcs += 1.0
                else:
                    self.A.edges_error_table[(p_p,c_p)]=EDGE_TYPE_ERROR
            else:
                self.A.edges_error_table[(p_p,c_p)]=EDGE_MATCH_ERROR

        #num_parsed_tags = len([i for i in visited_nodes if re.match('\w+-\d+',self.A.get_node_tag(i))])
        #num_gold_tags = len([j for j in self.gold_graph.nodes if re.match('\w+-\d+',self.gold_graph.get_node_tag(j))])
        #num_parsed_tags = len([i for i in visited_nodes if isinstance(self.A.get_node_tag(i),(ETag,ConstTag))])
        #num_gold_tags = len([j for j in self.gold_graph.nodes if isinstance(self.gold_graph.get_node_tag(j),(ETag,ConstTag))])
        #num_parsed_tags = len([i for i in visited_nodes if not (isinstance(self.A.get_node_tag(i),(ETag,ConstTag)) or re.match('\w+-\d+',self.A.get_node_tag(i)))])
        #num_gold_tags = len([j for j in self.gold_graph.nodes if not (isinstance(self.gold_graph.get_node_tag(j),(ETag,ConstTag)) or re.match('\w+-\d+',self.gold_graph.get_node_tag(j)))])
        num_parsed_tags = len(visited_nodes)
        num_gold_tags = len(self.gold_graph.nodes)
        return num_correct_labeled_arcs,num_correct_arcs,num_parsed_arcs,num_gold_arcs,num_correct_tags,num_parsed_tags,num_gold_tags
    '''
    def evaluate_actions(self,gold_state):
        gold_act_seq = gold_state.action_history
        parsed_act_seq = self.action_history
        confusion_matrix = np.zeros(shape=(len(GraphState.action_table),len(GraphState.action_table)))
        edge_label_count = defaultdict(float)
        # chop out the longer one
        common_step = len(gold_act_seq) if len(gold_act_seq) <= len(parsed_act_seq) else len(parsed_act_seq)
        for i in range(common_step):
            g_act = gold_act_seq[i]
            p_act = parsed_act_seq[i]

            confusion_matrix[g_act['type'],p_act['type']]+=1
            if g_act['type'] == p_act['type'] and g_act['type'] in ACTION_WITH_EDGE:
                if g_act == p_act:
                    edge_label_count[g_act['type']]+=1.0
        #for j in range(confusion_matrix.shape[0]):
        #    if j in ACTION_WITH_EDGE:
        #        confusion_matrix[j,j] = edge_label_count[j]/confusion_matrix[j,j] if confusion_matrix[j,j] != 0.0 else 0.0

        return confusion_matrix
    '''

    def get_score(self, action_obj, feature, train=True):
        act_type = action_obj['type'];
        act_idx = GraphState.model.class_codebook.get_index(act_type)[0];
        feat_idx, val_arr = zip(*map(GraphState.model.feature_codebook[act_idx].get_index,feature));
        #print("feat_idx is : " + str(feat_idx));
        feat_idx = list(feat_idx);
        val_arr = list(val_arr);

        #W2V_OFFSET_S0 = 0;
        #W2V_OFFSET_A = 300;
        #W2V_OFFSET_B = 600;
        #W2V_PROD_AB = 900;
        #W2V_PROD_AS = 1200;
        #W2V_PROD_SB = 1500;
        #if W2V_OFFSET_S0 in feat_idx and W2V_OFFSET_A in feat_idx:
        #    for i in range(0, W2V_LENGTH):
        #        feat_idx.append(W2V_PROD_AS + i);
        #        _a = val_arr[feat_idx.index(W2V_OFFSET_S0 + i)];
        #        _b = val_arr[feat_idx.index(W2V_OFFSET_A + i)];
        #        val_arr.append(abs(_a-_b));
        #if W2V_OFFSET_S0 in feat_idx and W2V_OFFSET_B in feat_idx:
        #    for i in range(0, W2V_LENGTH):
        #        feat_idx.append(W2V_PROD_SB + i);
        #        _a = val_arr[feat_idx.index(W2V_OFFSET_S0 + i)];
        #        _b = val_arr[feat_idx.index(W2V_OFFSET_B + i)];
        #        val_arr.append(abs(_a-_b));
        #if W2V_OFFSET_A in feat_idx and W2V_OFFSET_B in feat_idx:
        #    for i in range(0, W2V_LENGTH):
        #        feat_idx.append(W2V_PROD_AB + i);
        #        _a = val_arr[feat_idx.index(W2V_OFFSET_A + i)];
        #        _b = val_arr[feat_idx.index(W2V_OFFSET_B + i)];
        #        val_arr.append(abs(_a-_b));

        feat_idx_NN = [];
        val_arr_NN = [];
        for feat_idx_I in range(0, len(feat_idx)):
            if feat_idx[feat_idx_I] is not None:
                feat_idx_NN.append(feat_idx[feat_idx_I]);
                val_arr_NN.append(val_arr[feat_idx_I]);
        ## try to do the update catchup
        w_z = GraphState.model.weight[act_idx][feat_idx_NN];
        q_z = GraphState.model.q[act_idx][feat_idx_NN];
        num = np.sqrt(q_z);
        denom = np.copy(num);
        denom = denom + GraphState.model.eta*GraphState.model.C;
        quotient = np.divide(num, denom);
        nupdates = GraphState.num_updates - GraphState.model.u[act_idx][feat_idx_NN];
        #print("NUPDATES: " + str(nupdates));
        topow = np.power(quotient, nupdates);
        new_w = topow*w_z;

        GraphState.model.weight[act_idx][feat_idx_NN] = new_w;
        GraphState.model.u[act_idx][feat_idx_NN] = GraphState.num_updates;

        weight = GraphState.model.weight[act_idx];
        #print("weight shape: " + str(weight.shape));
        #print("weight: " + str(weight));

        fidx_widx = 0;
        score = weight[ [i for i in feat_idx if i is not None] ];
        #for feat_ in feat_idx_NN:
        #    if feat_ < CATEGORICAL_OFFSET:
        #        #print("the feat_ index is: " + str(feat_));
        #        #print("weight fidx_widx: " + str(weight[fidx_widx]));
        #        #print("val arr @ fidx_widx: " + str(val_arr_NN[fidx_widx]));
        #        score[fidx_widx] *= val_arr_NN[fidx_widx];
        #    else:
        #        break;
        #    fidx_widx += 1;

        score = np.sum(score, axis=0);
        #print("Scores array being returned from graphstate is: " + str(score));
        return score;

    def make_feat(self,action):
        feat = GraphState.model.feats_generator(self,action)
        return feat

    def get_current_node(self):
        return self.A.nodes[self.idx]

    def get_current_child(self):
        if self.cidx and self.cidx in self.A.nodes:
            return self.A.nodes[self.cidx]
        else:
            return None

    def apply(self,action):
        action_type = action['type']
        other_params = dict([(k,v) for k,v in action.items() if k!='type' and v is not None])
        self.action_history.append(action)
        return getattr(self,GraphState.action_table[action_type])(**other_params)


    def next1(self, edge_label=None):
        newstate = self.pcopy()
        if edge_label and edge_label is not START_EDGE:newstate.A.set_edge_label(newstate.idx,newstate.cidx,edge_label)
        newstate.beta.pop()
        newstate.cidx = newstate.beta.top() if newstate.beta else None
        #newstate.action_history.append(NEXT1)

        return newstate

    def next2(self, tag=None):
        newstate = self.pcopy()
        if tag: newstate.A.set_node_tag(newstate.idx,tag)
        newstate.sigma.pop()
        newstate.idx = newstate.sigma.top()
        newstate.beta = Buffer(newstate.A.nodes[newstate.idx].children) if newstate.idx != -1 else None
        if newstate.beta is not None: newstate.beta.push(START_ID)
        newstate.cidx = newstate.beta.top() if newstate.beta else None
        #newstate.action_history.append(NEXT2)

        return newstate

    def delete_node(self):
        newstate = self.pcopy()
        newstate.A.remove_node(newstate.idx,RECORD=True)
        newstate.sigma.pop()
        newstate.idx = newstate.sigma.top()
        newstate.beta = Buffer(newstate.A.nodes[newstate.idx].children) if newstate.idx != -1 else None
        if newstate.beta is not None: newstate.beta.push(START_ID)
        newstate.cidx = newstate.beta.top() if newstate.beta else None
        #newstate.action_history.append(DELETENODE)

        return newstate

    def infer(self, tag):
        '''
        infer abstract node on core noun
        '''
        newstate = self.pcopy()
        abt_node_index = newstate.A.new_abt_node(newstate.idx,tag)

        # add the atomic info from its core noun
        #abt_atomics = {}
        #abt_atomics['id'] = abt_node_index
        #abt_atomics['form'] = ABT_FORM
        #abt_atomics['lemma'] = ABT_LEMMA
        #abt_atomics['pos'] = GraphState.sent[newstate.idx]['pos'] if isinstance(newstate.idx,int) else GraphState.abt_tokens[newstate.idx]['pos']
        #abt_atomics['ne'] = GraphState.sent[newstate.idx]['ne'] if isinstance(newstate.idx,int) else GraphState.abt_tokens[newstate.idx]['ne']
        #abt_atomics['rel'] = GraphState.sent[newstate.idx]['rel'] if isinstance(newstate.idx,int) else GraphState.abt_tokens[newstate.idx]['rel']
        #GraphState.abt_tokens[abt_node_index] = abt_atomics

        tmp = newstate.sigma.pop()
        newstate.sigma.push(abt_node_index)
        newstate.sigma.push(tmp)
        return newstate
    '''
    infer abstract node on edge pair: may cause feature inconsistency
    def infer1(self):
        newstate = self.pcopy()
        abt_node_index = newstate.A.new_abt_node(newstate.idx)
        newstate.A.reattach_node(newstate.idx,newstate.cidx,abt_node_index,NULL_EDGE)
        tmp = newstate.sigma.pop()
        newstate.sigma.push(abt_node_index)
        newstate.sigma.push(tmp)
        newstate.beta.pop()
        newstate.beta.append(abt_node_index)
        newstate.cidx = newstate.beta.top() if newstate.beta else None
        return newstate
    '''

    '''
    def delete_edge(self):
        newstate = self.pcopy()
        newstate.A.remove_edge(newstate.idx,newstate.cidx)
        #catm = self.atomics[newstate.cidx]
        #cparents = sorted(newstate.A.nodes[self.cidx].parents)
        #catm['blp1'] = GraphState.sent[cparents[0]] if cparents and cparents[0] < self.cidx else NOT_ASSIGNED
        newstate.beta.pop()
        newstate.cidx = newstate.beta.top() if newstate.beta else None
        #newstate.action_history.append(DELETEEDGE)

        return newstate
    '''

    def reattach(self,parent_to_attach=None,edge_label=None):
        newstate = self.pcopy()
        newstate.A.reattach_node(newstate.idx,newstate.cidx,parent_to_attach,edge_label)
        newstate.beta.pop()
        newstate.cidx = newstate.beta.top() if newstate.beta else None
        return newstate

    def swap(self,edge_label):
        newstate = self.pcopy()
        newstate.A.swap_head2(newstate.idx,newstate.cidx,newstate.sigma,edge_label)
        newstate._fix_prop_feature(newstate.idx,newstate.cidx)
        #newstate.idx = newstate.cidx
        tmp = newstate.sigma.pop()
        tmp1 = newstate.sigma.pop() if newstate.A.nodes[tmp].num_parent_infer > 0 else None
        if newstate.cidx not in newstate.sigma: newstate.sigma.push(newstate.cidx)
        if tmp1: newstate.sigma.push(tmp1)
        newstate.sigma.push(tmp)

        # TODO revisit
        #newstate.beta.pop()
        newstate.beta = Buffer([c for c in newstate.A.nodes[newstate.idx].children if c != newstate.cidx and c not in newstate.A.nodes[newstate.cidx].parents])
        newstate.cidx = newstate.beta.top() if newstate.beta else None
        #newstate.action_history.append(SWAP)

        return newstate
    '''
    def change_head(self,goldParent):
        newstate = self.pcopy()
        newstate.A.remove_edge(newstate.idx,newstate.cidx)
        newstate.A.add_edge(goldParent,newstate.cidx)
        newstate.A.relativePos(newstate.cidx,goldParent)
    '''

    def reentrance(self,parent_to_add,edge_label=None):
        newstate = self.pcopy()
        #delnodes = newstate.A.clear_up(parent_to_add,newstate.cidx)
        #for dn in delnodes:
        #    if dn in newstate.sigma:
        #        newstate.sigma.remove(dn)
        if edge_label:
            try:
                newstate.A.add_edge(parent_to_add,newstate.cidx,edge_label)
            except KeyError:
                import pdb
                pdb.set_trace()
        else:
            newstate.A.add_edge(parent_to_add,newstate.cidx)
        return newstate

    def add_child(self,child_to_add,edge_label=None):
        newstate = self.pcopy()
        if edge_label:
            newstate.A.add_edge(newstate.idx,child_to_add,edge_label)
        else:
            newstate.A.add_edge(newstate.idx,child_to_add)


        #hoffset,voffset = GraphState.deptree.relativePos(newstate.idx,node_to_add)
        #atype = GraphState.deptree.relativePos2(newstate.idx,child_to_add)
        #self.new_actions.add('add_child_('+str(hoffset)+')_('+str(voffset)+')_'+str(GraphState.sentID))
        #self.new_actions.add('add_child_%s_%s'%(atype,str(GraphState.sentID)))
        #newstate.action_history.append(ADDCHILD)
        return newstate


    def _fix_prop_feature(self,idx,cidx):
        '''update cidx's prop feature with idx's prop feature'''
        if isinstance(idx,int) and isinstance(cidx,int):
            ctok = GraphState.sent[cidx]
            tok = GraphState.sent[idx]
            ctok['pred'] = ctok.get('pred',{})
            ctok['pred'].update(dict((k,v) for k,v in tok.get('pred',{}).items() if k!=cidx))
            for prd in tok.get('pred',{}).copy():
                if prd != cidx:
                    try:
                        tmp = GraphState.sent[prd]['args'].pop(idx)
                        GraphState.sent[idx]['pred'].pop(prd)
                    except KeyError:
                        import pdb
                        pdb.set_trace()
                    GraphState.sent[prd]['args'][cidx] = tmp


            ctok['args'] = ctok.get('args',{})
            ctok['args'].update(dict((k,v) for k,v in tok.get('args',{}).items() if k!=cidx))
            for arg in tok.get('args',{}).copy():
                if arg != cidx:
                    try:
                        atmp = GraphState.sent[arg]['pred'].pop(idx)
                        GraphState.sent[idx]['args'].pop(arg)
                    except KeyError:
                        import pdb
                        pdb.set_trace()
                    GraphState.sent[arg]['pred'][cidx] = atmp

    def replace_head(self):
        """
        Use current child to replace current node
        """
        newstate = self.pcopy()
        newstate.beta = Buffer([c for c in newstate.A.nodes[newstate.idx].children if c != newstate.cidx and c not in newstate.A.nodes[newstate.cidx].parents])
        #for old_c in newstate.A.nodes[newstate.cidx].children: newstate.beta.push(old_c)
        newstate.A.replace_head(newstate.idx,newstate.cidx)
        newstate._fix_prop_feature(newstate.idx,newstate.cidx)
        if newstate.idx in newstate.sigma: newstate.sigma.remove(newstate.idx)
        if newstate.cidx in newstate.sigma: newstate.sigma.remove(newstate.cidx) # pushing cidx to top
        newstate.sigma.push(newstate.cidx)
        newstate.A.record_rep_head(newstate.cidx,newstate.idx)
        newstate.idx = newstate.cidx
        newstate.cidx = newstate.beta.top() if newstate.beta else None
        #newstate.action_history.append(REPLACEHEAD)

        return newstate

    def merge(self):
        """
        merge nodes to form entity
        """
        newstate = self.pcopy()
        tmp1 = newstate.idx
        tmp2 = newstate.cidx
        #try:
        newstate.A.merge_node(tmp1,tmp2)
        #except KeyError:
        #    import pdb
        #    pdb.set_trace()

        if tmp1 < tmp2:
            if tmp2 in newstate.sigma:
                newstate.sigma.remove(tmp2)
        else:
            if tmp2 in newstate.sigma: newstate.sigma.remove(tmp2) # pushing tmp2 to the top
            newstate.sigma.push(tmp2)
            if tmp1 in newstate.sigma: newstate.sigma.remove(tmp1)


        newstate.idx = tmp1 if tmp1 < tmp2 else tmp2
        newstate.cidx = tmp2 if tmp1 < tmp2 else tmp1
        GraphState.sent[newstate.idx]['rel'] = GraphState.sent[tmp1]['rel']
        newstate._fix_prop_feature(newstate.cidx,newstate.idx)
        #newstate.A.merge_node(newstate.idx,newstate.cidx)
        newstate.beta = Buffer(newstate.A.nodes[newstate.idx].children[:])
        newstate.cidx = newstate.beta.top() if newstate.beta else None
        #newstate.action_history.append(MERGE)

        return newstate

    @staticmethod
    def get_parsed_amr(span_graph):

        def unpack_node(node,amr,variable):
            node_id = node.start
            node_tag = node.tag
            #if node.tag is None:
            #    import pdb
            #    pdb.set_trace()
            core_var = None
            tokens_in_span = GraphState.sent[node.start:node.end] if isinstance(node_id,int) else node.words
            if isinstance(node_tag,ETag):
                # normalize country adjective

                foo = amr[variable]
                pre_abs_id = None
                rel = None
                for i,abs_tag in enumerate(node_tag.split('+')):

                    if i == 0: # node already initialized
                        if '@' in abs_tag:
                            #abs_tag,rel = abs_tag.split('@')
                            raise Exception('Tag format error')
                        amr.node_to_concepts[variable] = abs_tag
                        pre_abs_id = variable
                    elif abs_tag == '-': # negation
                        abs_id = Polarity(abs_tag)
                        foo = amr[abs_id]
                        rel = 'polarity'
                        amr._add_triple(pre_abs_id,rel,abs_id)
                        pre_abs_id = abs_id
                    else:

                        if '@' in abs_tag:
                            #abs_tag,rel = abs_tag.split('@')
                            rel, sub_abs_tag = abs_tag.split('@')

                            abs_id = sub_abs_tag[0].lower()
                            j = 0
                            while abs_id in amr:
                                j+=1
                                abs_id = abs_id[0]+str(j)

                            foo = amr[abs_id]
                            amr._add_triple(pre_abs_id,rel,abs_id)
                            amr.node_to_concepts[abs_id] = sub_abs_tag
                            pre_abs_id = abs_id
                        else:
                            pass # duplicate concept
                            #rel = None
                            #rel = abs_tag




                last_abs_id = pre_abs_id
                last_abs_tag = abs_tag if abs_tag else node_tag.split('+')[i-1]
                if last_abs_tag == '-' or '@' in abs_tag:
                    return variable,core_var

                rel_in_span = 'op' if rel is None else rel
                for i,tok in enumerate(tokens_in_span):
                    ###########
                    tok['form'] = tok['form'].replace(':','-') # handle format exceptions
                    if node_tag == 'country+name@name+name':
                        if tok['form'] in COUNTRY_LIST:
                            tok['form'] = COUNTRY_LIST[tok['form']]

                    ###########

                    foo = amr[tok['form']]

                    if last_abs_tag == 'name':
                        amr._add_triple(last_abs_id,'op'+str(i+1),StrLiteral(tok['form']))
                    elif last_abs_tag == 'date-entity':
                        date_pattern = [
                            ('d1','^({0}{0}{0}{0})(\-{0}{0})?(\-{0}{0})?$'.format('[0-9]')),
                            ('d2','^({0}{0})({0}{0})({0}{0})$'.format('[0-9]'))
                        ]
                        date_rule = '|'.join('(?P<%s>%s)'%(p,d) for p,d in date_pattern)
                        m = re.match(date_rule,tok['form'])
                        if m:
                            year,month,day = None,None,None
                            date_type = m.lastgroup

                            if date_type == 'd1':
                                year = m.group(2)
                                if m.group(3) is not None: month = str(int(m.group(3)[1:]))
                                if m.group(4) is not None: day = str(int(m.group(4)[1:]))
                            elif date_type == 'd2':
                                year = '20'+m.group(6)
                                month = str(int(m.group(7)))
                                day = str(int(m.group(8)))
                            else:
                                #raise ValueError('undefined date pattern')
                                pass

                            foo = amr[year]
                            amr._add_triple(last_abs_id,'year',Quantity(year))
                            if month and month != '0':
                                foo = amr[month]
                                amr._add_triple(last_abs_id,'month',Quantity(month))
                            if day and day != '0':
                                foo = amr[day]
                                amr._add_triple(last_abs_id,'day',Quantity(day))
                    elif last_abs_tag.endswith('-quantity'):
                        new_id = tok['form'][0].lower()
                        j = 0
                        while new_id in amr:
                            j+=1
                            new_id = new_id[0]+str(j)

                        foo = amr[new_id]
                        amr.node_to_concepts[new_id] = tok['form']
                        amr._add_triple(last_abs_id,'unit',new_id)
                    elif last_abs_tag == 'have-org-role-91':
                        new_id = tok['lemma'][0].lower()
                        j = 0
                        while new_id in amr:
                            j+=1
                            new_id = new_id[0]+str(j)

                        foo = amr[new_id]
                        core_var = new_id
                        amr.node_to_concepts[new_id] = tok['lemma'].lower()
                        amr._add_triple(last_abs_id,rel_in_span,new_id)

                    else:
                        if re.match('[0-9\-]+',tok['form']):
                            amr._add_triple(last_abs_id,rel_in_span,Quantity(tok['form']))
                        else:
                            new_id = tok['lemma'][0].lower()
                            j = 0
                            while new_id in amr:
                                j+=1
                                new_id = new_id[0]+str(j)

                            foo = amr[new_id]
                            amr.node_to_concepts[new_id] = tok['lemma'].lower()
                            amr._add_triple(last_abs_id,rel_in_span,new_id)
            elif isinstance(node_tag,ConstTag):
                foo = amr[node_tag]
                variable = node_tag
            else:
                if r'/' in node_tag:
                    #import pdb
                    #pdb.set_trace()
                    variable = StrLiteral(node_tag)
                    foo = amr[variable]
                else:
                    foo = amr[variable]
                    node_tag = node_tag.replace(':','-')
                    amr.node_to_concepts[variable] = node_tag # concept tag

            return variable,core_var

        amr = AMR()
        span_graph.flipConst()
        node_prefix = 'x'
        cpvar_cache = {}

        for parent,child in span_graph.tuples():
            pvar = node_prefix+str(parent)
            cvar = node_prefix+str(child)


            try:
                if parent == 0:
                    if cvar not in amr:
                        cvar,ccvar = unpack_node(span_graph.nodes[child],amr,cvar)
                        cpvar_cache[cvar] = ccvar
                    if cvar not in amr.roots: amr.roots.append(cvar)
                else:
                    rel_label = span_graph.get_edge_label(parent,child)
                    if pvar not in amr:
                        pvar,cpvar = unpack_node(span_graph.nodes[parent],amr,pvar)
                        cpvar_cache[pvar]=cpvar
                    if cvar not in amr:
                        cvar,ccvar = unpack_node(span_graph.nodes[child],amr,cvar)
                        cpvar_cache[cvar]=ccvar
                    if cpvar_cache.get(pvar,None) and rel_label == 'mod':
                        amr._add_triple(cpvar_cache[pvar],rel_label,cvar)
                    else:
                        amr._add_triple(pvar,rel_label,cvar)
            except ValueError as e:
                print e
                #print span_graph.graphID

        if len(amr.roots) > 1:
            foo =  amr[FAKE_ROOT_VAR]
            amr.node_to_concepts[FAKE_ROOT_VAR] = FAKE_ROOT_CONCEPT
            for multi_root in amr.roots:
                amr._add_triple(FAKE_ROOT_VAR,FAKE_ROOT_EDGE,multi_root)
            amr.roots = [FAKE_ROOT_VAR]
        elif len(amr.roots) == 0 and len(amr.keys()) != 0:
            foo =  amr[FAKE_ROOT_VAR]
            amr.node_to_concepts[FAKE_ROOT_VAR] = FAKE_ROOT_CONCEPT
            for mlt_root in span_graph.get_multi_roots():
                mrvar = node_prefix + str(mlt_root)
                if mrvar in amr:
                    amr._add_triple(FAKE_ROOT_VAR,FAKE_ROOT_EDGE,mrvar)
            amr.roots=[FAKE_ROOT_VAR]
        elif len(amr.roots) == 1 and amr.roots[0] not in amr.node_to_concepts: # Const tag
            foo =  amr[FAKE_ROOT_VAR]
            amr.node_to_concepts[FAKE_ROOT_VAR] = FAKE_ROOT_CONCEPT
            amr._add_triple(FAKE_ROOT_VAR,FAKE_ROOT_EDGE,amr.roots[0])
            amr.roots = [FAKE_ROOT_VAR]
        elif len(amr.keys()) == 0:
            foo =  amr[FAKE_ROOT_VAR]
            amr.node_to_concepts[FAKE_ROOT_VAR] = FAKE_ROOT_CONCEPT
            for mlt_root in span_graph.get_multi_roots():
                mrvar = node_prefix + str(mlt_root)
                foo = amr[mrvar]
                amr.node_to_concepts[mrvar] = span_graph.nodes[mlt_root].tag
                amr._add_triple(FAKE_ROOT_VAR,FAKE_ROOT_EDGE,mrvar)
            amr.roots=[FAKE_ROOT_VAR]

        else:
            pass


        return amr


    def print_config(self, column_len = 80):
        output = ''
        if self.cidx:
            if self.idx == START_ID:
                span_g = START_FORM
            else:
                span_g = ','.join(tok['form'] for tok in GraphState.sent[self.idx:self.A.nodes[self.idx].end]) if isinstance(self.idx,int) else ','.join(self.A.nodes[self.idx].words)
            if self.cidx == START_ID:
                span_d = START_FORM
            else:
                span_d = ','.join(tok['form'] for tok in GraphState.sent[self.cidx:self.A.nodes[self.cidx].end]) if isinstance(self.cidx,int) else ','.join(self.A.nodes[self.cidx].words)
            output += 'ID:%s %s\nParent:(%s-%s) Child:(%s-%s)'%(str(GraphState.sentID),self.text,\
                                                   span_g, self.idx, \
                                                   span_d, self.cidx)
        else:
            '''
            if self.action_history and self.action_history[-1] == ADDCHILD: # add child
                added_child_idx = self.A.nodes[self.idx].children[-1]
                output += 'ID:%s %s\nParent:(%s-%s) add child:(%s-%s)'%(str(GraphState.sentID),self.text,\
                                                           ','.join(tok['form'] for tok in GraphState.sent[self.idx:self.A.nodes[self.idx].end]), self.idx, \
                                                    ','.join(tok['form'] for tok in GraphState.sent[added_child_idx:self.A.nodes[added_child_idx].end]), added_child_idx)
            else:
            '''
            if self.idx == START_ID:
                span_g = START_FORM
                output += 'ID:%s %s\nParent:(%s-%s) Children:%s'%(str(GraphState.sentID),self.text,\
                                                                      span_g, self.idx, 'None')
            else:
                span_g = ','.join(tok['form'] for tok in GraphState.sent[self.idx:self.A.nodes[self.idx].end]) if isinstance(self.idx,int) else ','.join(self.A.nodes[self.idx].words)
                output += 'ID:%s %s\nParent:(%s-%s) Children:%s'%(str(GraphState.sentID),self.text,\
                                                                      span_g, self.idx, \
                                                                      ['('+','.join(tok['form'] for tok in GraphState.sent[c:self.A.nodes[c].end])+')' if isinstance(c,int) else '('+','.join(self.A.nodes[c].words)+')' for c in self.A.nodes[self.idx].children])

        output += '\n'
        parsed_tuples = self.A.tuples()
        ref_tuples = self.gold_graph.tuples()
        num_p = len(parsed_tuples)
        num_r = len(ref_tuples)
        tnum = num_r if num_r > num_p else num_p
        for i in range(tnum):
            strformat = '{0:<%s}|{1:<%s}' % (column_len,column_len)
            if i < num_p and i < num_r:
                g,d = parsed_tuples[i]
                gg,gd = ref_tuples[i]
                parsed_edge_label = self.A.get_edge_label(g,d)
                gold_edge_label = self.gold_graph.get_edge_label(gg,gd)
                gold_span_gg = ','.join(tok['form'] for tok in GraphState.sent[gg:self.gold_graph.nodes[gg].end]) if isinstance(gg,int) else ','.join(self.gold_graph.nodes[gg].words)
                gold_span_gd = ','.join(tok['form'] for tok in GraphState.sent[gd:self.gold_graph.nodes[gd].end]) if isinstance(gd,int) else ','.join(self.gold_graph.nodes[gd].words)
                parsed_span_g = ','.join(tok['form'] for tok in GraphState.sent[g:self.A.nodes[g].end]) if isinstance(g,int) else ','.join(self.A.nodes[g].words)
                parsed_span_d = ','.join(tok['form'] for tok in GraphState.sent[d:self.A.nodes[d].end]) if isinstance(d,int) else ','.join(self.A.nodes[d].words)
                parsed_tag_g = self.A.get_node_tag(g)
                parsed_tag_d = self.A.get_node_tag(d)
                gold_tag_gg = self.gold_graph.get_node_tag(gg)
                gold_tag_gd = self.gold_graph.get_node_tag(gd)
                parsed_tuple_str = "(%s(%s-%s:%s),(%s-%s:%s))" % (parsed_edge_label, parsed_span_g, g, parsed_tag_g, parsed_span_d, d, parsed_tag_d)
                ref_tuple_str = "(%s(%s-%s:%s),(%s-%s:%s))" % (gold_edge_label, gold_span_gg, gg, gold_tag_gg, gold_span_gd, gd, gold_tag_gd)
                output += strformat.format(parsed_tuple_str,ref_tuple_str)
                output += '\n'
            elif i < num_p and i >= num_r:
                g,d = parsed_tuples[i]
                parsed_edge_label = self.A.get_edge_label(g,d)
                parsed_tag_g = self.A.get_node_tag(g)
                parsed_tag_d = self.A.get_node_tag(d)
                parsed_span_g = ','.join(tok['form'] for tok in GraphState.sent[g:self.A.nodes[g].end]) if isinstance(g,int) else ','.join(self.A.nodes[g].words)
                parsed_span_d = ','.join(tok['form'] for tok in GraphState.sent[d:self.A.nodes[d].end]) if isinstance(d,int) else ','.join(self.A.nodes[d].words)
                parsed_tuple_str = "(%s(%s-%s:%s),(%s-%s:%s))" % (parsed_edge_label, parsed_span_g, g, parsed_tag_g, parsed_span_d, d, parsed_tag_d)
                output += strformat.format(parsed_tuple_str,'*'*column_len)
                output += '\n'
            elif i >= num_p and i < num_r:
                gg,gd = ref_tuples[i]
                gold_edge_label = self.gold_graph.get_edge_label(gg,gd)
                gold_span_gg = ','.join(tok['form'] for tok in GraphState.sent[gg:self.gold_graph.nodes[gg].end]) if isinstance(gg,int) else ','.join(self.gold_graph.nodes[gg].words)
                gold_span_gd = ','.join(tok['form'] for tok in GraphState.sent[gd:self.gold_graph.nodes[gd].end]) if isinstance(gd,int) else ','.join(self.gold_graph.nodes[gd].words)
                gold_tag_gg = self.gold_graph.get_node_tag(gg)
                gold_tag_gd = self.gold_graph.get_node_tag(gd)
                ref_tuple_str = "(%s(%s-%s:%s),(%s-%s:%s))" % (gold_edge_label, gold_span_gg, gg, gold_tag_gg, gold_span_gd, gd, gold_tag_gd)
                output += strformat.format('*'*column_len,ref_tuple_str)
                output += '\n'
            else:
                pass

        return output


    def write_basic_amr(self,out,CONST_REL='ARG0'):
        '''
        this method takes the unlabeled edges produced by the parser and
        adds them with fake amr relation which is mapped from dependency tag set
        '''
        CoNLLSent = GraphState.sent
        parsed_tuples = self.A.tuples()
        out.write(str(GraphState.sentID)+'\n')
        fake_amr_triples = []
        for g,d in parsed_tuples:
            gov = CoNLLSent[g]
            dep = CoNLLSent[d]
            if dep['head'] == gov['id']: # tuple also in dependency tree
                rel = get_fake_amr_relation_mapping(dep['rel']) if get_fake_amr_relation_mapping(dep['rel']) != 'NONE' else CONST_REL
                fake_amr_triples.append((rel,gov['lemma'],dep['lemma']))
            else:
                fake_amr_triples.append((CONST_REL,gov['lemma'],dep['lemma']))
            out.write(str(fake_amr_triples[-1])+'\n')
        return fake_amr_triples
