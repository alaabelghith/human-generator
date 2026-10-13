"""Parameter interface of the Human Generator HDA (used by tools/build_hda.py)."""
import hou

F, I, M, T, S = hou.FloatParmTemplate, hou.IntParmTemplate, hou.MenuParmTemplate, hou.ToggleParmTemplate, hou.StringParmTemplate

# shared by the PythonCook section and the Generate button: find the package through $HUMAN_GENERATOR
BOOT = '''import os, sys, importlib
_root = os.environ.get("HUMAN_GENERATOR") or hou.getenv("HUMAN_GENERATOR")
if not _root:
    raise hou.NodeError("HUMAN_GENERATOR is not set. Run install.ps1, then restart Houdini.")
_p = os.path.join(_root, "houdini", "python")
if _p not in sys.path:
    sys.path.insert(0, _p)
import human_generator.client, human_generator.node
if os.environ.get("HG_DEV") == "1":
    importlib.reload(human_generator.client); importlib.reload(human_generator.node)
'''
COOK = BOOT + "human_generator.node.cook(hou.pwd())\n"
GENERATE = BOOT + 'human_generator.node.generate(kwargs["node"])\n'

ANNY_ONLY = "{ body_model == 0 } { body_model == soma }"   # hide when the SOMA backend is selected
SOMA_ONLY = "{ body_model == 1 } { body_model == anny }"


def parms():
    g = hou.ParmTemplateGroup()
    g.append(M("output", "Output", ("body", "skeleton", "both"), ("Body", "Skeleton", "Body + Skeleton")))
    g.append(T("smooth_normals", "Smooth Normals (shading)", default_value=True))
    g.append(I("smooth_iter", "Smooth Surface Iterations", 1, default_value=(0,), min=0, max=50,
               help="Relaxes the mesh like a Smooth SOP (removes folds under breasts and belly) and keeps the volume. 0 = off."))
    g.append(F("smooth_strength", "Smooth Strength", 1, default_value=(0.5,), min=0.0, max=1.0))

    b = hou.FolderParmTemplate("f_body", "Body")
    b.addParmTemplate(M("body_model", "Body Model", ("soma", "anny"),
                        ("SOMA PCA (neutral, 12 shape components)", "Anny (age, height, weight, gender, muscle...)"), default_value=1))
    b.addParmTemplate(F("height", "Global Scale", 1, default_value=(1.0,), min=0.5, max=1.5))
    anny = [I("age_years", "Age (years)", 1, default_value=(30,), min=1, max=100, min_is_strict=True, max_is_strict=True),
            I("height_cm", "Height (cm)", 1, default_value=(178,), min=50, max=250, min_is_strict=True, max_is_strict=True),
            I("weight_kg", "Weight (kg)", 1, default_value=(75,), min=10, max=170, min_is_strict=True, max_is_strict=True,
              help="Reached exactly with Anny's weight phenotype plus a solved fat amount. Measured value: detail attribute weight_kg."),
            F("ph_gender", "Gender (0 male - 1 female)", 1, default_value=(0.0,), min=0.0, max=1.0, min_is_strict=True, max_is_strict=True),
            F("ph_muscle", "Muscle", 1, default_value=(0.5,), min=0.0, max=1.0, min_is_strict=True, max_is_strict=True),
            F("ph_proportions", "Proportions (0 ideal - 1 uncommon)", 1, default_value=(0.5,), min=0.0, max=1.0,
              min_is_strict=True, max_is_strict=True, help="0 = idealised proportions, 1 = irregular, everyday proportions."),
            F("ph_cupsize", "Breast Size", 1, default_value=(0.5,), min=0.0, max=1.0, min_is_strict=True, max_is_strict=True),
            F("ph_firmness", "Breast Firmness", 1, default_value=(0.5,), min=0.0, max=1.0, min_is_strict=True, max_is_strict=True,
              help="Values above ~0.9 together with a large size can crease under the breast."),
            F("buttocks", "Buttocks Size", 1, default_value=(0.0,), min=-1.0, max=1.0,
              help="Anny local change 'buttocks-volume-incr': negative = smaller, positive = bigger."),
            S("anny_local", "Anny Local Changes", 1, default_value=("",),
              help="Optional fine edits, e.g.  torso-vshape-incr=0.5, hip-scale-horiz-incr=-0.3  (139 names, 0 = none).")]
    for pt in anny:
        pt.setConditional(hou.parmCondType.HideWhen, ANNY_ONLY)
        b.addParmTemplate(pt)
    soma = [I("rand_seed", "Random Seed", 1, default_value=(0,), min=0, max=999),
            F("rand_amount", "Random Amount", 1, default_value=(0.0,), min=0.0, max=2.0)]
    soma += [F("pc_%d" % i, "Shape PC %d" % (i + 1), 1, default_value=(0.0,), min=-3.0, max=3.0) for i in range(12)]
    soma += [F(p, lb, 1, default_value=(1.0,), min=0.7, max=1.3) for p, lb in (
        ("sc_upperarm", "Upper Arm Length"), ("sc_forearm", "Forearm Length"), ("sc_hand", "Hand Length"),
        ("sc_fingers", "Finger Length"), ("sc_shin", "Shin Length"), ("sc_foot", "Foot Length"))]
    for pt in soma:
        pt.setConditional(hou.parmCondType.HideWhen, SOMA_ONLY)
        b.addParmTemplate(pt)

    e = hou.FolderParmTemplate("f_eyes", "Eyes")
    e.addParmTemplate(T("eyes", "Add Eyeballs", default_value=True,
                        help="Two eyeballs on the eye joints. Groups: eyes, left_eye, right_eye, iris, pupil."))

    m = hou.FolderParmTemplate("f_motion", "Motion")
    m.addParmTemplate(T("use_motion", "Use Motion", default_value=True))
    m.addParmTemplate(S("prompt", "Prompt", 1,
                        default_value=("a person walks forward confidently. stops and looks around. waves with the right hand.",),
                        help="One or several sentences. Each sentence is one segment of the motion, blended into the next."))
    m.addParmTemplate(S("durations", "Segment Durations (s)", 1, default_value=("3 2 2",),
                        help="One duration per sentence, e.g. '3 2 2'. Empty = every sentence uses Duration."))
    m.addParmTemplate(F("duration", "Duration (s)", 1, default_value=(4.0,), min=1.0, max=10.0))
    m.addParmTemplate(I("transition", "Transition Frames", 1, default_value=(5,), min=1, max=30))
    m.addParmTemplate(I("seed", "Seed", 1, default_value=(1,), min=0, max=999))
    m.addParmTemplate(T("follow_path", "Follow Path (input curve)", default_value=True,
                        help="Wire a curve into the node's input: Kimodo generates the motion on that path."))
    m.addParmTemplate(S("path_share", "Path Share per Sentence", 1, default_value=("",),
                        help="Fraction of the curve each sentence covers, e.g. '0.85 0.15'. Empty = proportional to the durations."))
    m.addParmTemplate(T("path_reverse", "Reverse Path Direction", default_value=False))
    btn = hou.ButtonParmTemplate("generate", "Generate Motion")
    btn.setScriptCallback(GENERATE)
    btn.setScriptCallbackLanguage(hou.scriptLanguage.Python)
    m.addParmTemplate(btn)
    m.addParmTemplate(hou.SeparatorParmTemplate("motion_sep"))
    m.addParmTemplate(S("motion", "Motion File (Kimodo .npz)", 1, default_value=("",), string_type=hou.stringParmType.FileReference))
    m.addParmTemplate(T("inplace", "In Place (no forward travel)", default_value=False))
    m.addParmTemplate(T("scale_stride", "Scale Stride to Body (no foot sliding)", default_value=True))
    m.addParmTemplate(I("start_frame", "Start Frame", 1, default_value=(1,), min=1, max=500))
    m.addParmTemplate(T("loop", "Loop", default_value=False))

    s = hou.FolderParmTemplate("f_settings", "Settings")
    s.addParmTemplate(S("server", "Server", 1, default_value=("http://127.0.0.1:8777",)))
    s.addParmTemplate(M("lod", "Body LOD", ("mid", "low", "xlo"), ("Mid (18k points)", "Low (4.5k)", "Extra Low (612)")))
    s.addParmTemplate(T("ground", "Feet on Ground", default_value=True))
    for f in (b, e, m, s):
        g.append(f)
    return g
