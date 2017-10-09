class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class feature_constraints:
    increment = ['count_acroform', 'count_image_xlarge', 'count_acroform_obs', 'count_image_xsmall', 'count_action',
                 'count_javascript', 'count_action_obs', 'count_javascript_obs', 'count_box_a4', 'count_js',
                 'count_box_legal', 'count_js_obs', 'count_box_letter', 'count_obj', 'count_box_other', 'count_objstm',
                 'count_box_overlap', 'count_objstm_obs', 'count_endobj', 'count_page', 'count_endstream',
                 'count_page_obs', 'count_eof', 'count_startxref', 'count_font', 'count_stream', 'count_font_obs',
                 'count_trailer', 'count_image_large', 'count_xref', 'count_image_med', 'size', 'count_image_small']

    incre_decre = ['author_dot', 'keywords_dot', 'subject_dot', 'author_lc', 'keywords_lc', 'subject_lc', 'author_num',
                   'keywords_num', 'subject_num', 'author_oth', 'keywords_oth', 'subject_oth', 'author_uc',
                   'keywords_uc', 'subject_uc', 'createdate_ts', 'moddate_ts', 'title_dot', 'createdate_tz',
                   'moddate_tz', 'title_lc', 'creator_dot', 'producer_dot', 'title_num', 'creator_lc', 'producer_lc',
                   'title_oth', 'creator_num', 'producer_num', 'title_uc', 'creator_oth', 'producer_oth', 'version',
                   'creator_uc', 'producer_uc']
