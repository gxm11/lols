import pkg_resources
import sys

from .bo import BoMain
from .io import MainOutput
from .io import Settings
from .pp import PPmain
from .mep import MEPmain
from .utils import RstManager
from .utils import Timer


def main(args=None):
    """The main routine."""
    # start timers
    global_timer = Timer()
    local_timer = Timer()

    if args is None:
        args = sys.argv[1:]

    if not args_ok(args):  # INVALID ARGUMENTS -> END PROGRAM
        version = pkg_resources.require("laos-boats")[0].version
        print('BOSS version ' + str(version) + '\n' +
              'Usage:\n' +
              '   boss op <inputfile or rst-file>\n' +
              '   boss o <inputfile or rst-file>\n' +
              '   boss p <rst-file> <out-file>\n' +
              '   boss m <rst-file> <minima-file>\n' +
              '   boss s <inputfile>\n' +
              'See the documentation for further instructions.')
        return

    if not files_ok(args[1:]):  # INPUT FILE DOESN'T OPEN -> END PROGRAM
        return

    STS = Settings(args[1], global_timer)
    STS.doing_pp = True if 'p' in args[0] and len(STS.pp_iters) > 0 else False

    # don't overwrite an optimization run's outfile
    if len(args) == 3:
        ipt_outfile = args[2]
        if 'm' in args[0]:
            STS.outfile = STS.outfile[:-4] + '_mep.out'
        else:
            STS.outfile = STS.outfile[:-4] + '_pp.out'

    # initialize main output
    mainOutput = MainOutput(STS)

    if 'o' in args[0] and (STS.initpts + STS.iterpts) > 0:
        # run BO
        local_timer.startLap()
        rstManager = RstManager(STS)
        bo = BoMain(STS, mainOutput, rstManager)
        bo.run_optimization()
        mainOutput.progress_msg('| Bayesian optimization completed, ' +
                                'time [s] %s' % (local_timer.str_lapTime()),
                                1, True, True)

    if 'p' in args[0] and len(STS.pp_iters) > 0:
        # run post-processing
        local_timer.startLap()
        mainOutput.progress_msg('Starting post-processing...', 1, True)
        mainOutput.section_header("POST-PROCESSING")
        ipt_rstfile = STS.rstfile if 'o' in args[0] else args[1]
        if len(args) != 3:
            ipt_outfile = STS.outfile
        PPmain(STS, ipt_rstfile, ipt_outfile, mainOutput)
        mainOutput.progress_msg('Post-processing completed, ' +
                                'time [s] %s' % (local_timer.str_lapTime()), 1)

    if 'm' == args[0]:
        # find minimum energy paths
        local_timer.startLap()
        mainOutput.progress_msg('Finding minimum energy paths...', 1, True)
        mainOutput.section_header("MINIMUM ENERGY PATHS")
        MEPmain(STS, args[1], args[2], mainOutput)
        mainOutput.progress_msg('Finding minimum energy paths completed, ' +
                                'time [s] %s' % (local_timer.str_lapTime()), 1)

    if 's' == args[0] and (STS.initpts + STS.iterpts) > 0:
        # run BO
        local_timer.startLap()
        rstManager = RstManager(STS)
        bo = BoMain(STS, mainOutput, rstManager)
        bo.run_optimization_single()
        mainOutput.progress_msg('| Bayesian optimization completed, ' +
                                'time [s] %s' % (local_timer.str_lapTime()),
                                1, True, True)

    mainOutput.footer(global_timer.str_totalTime())


def args_ok(args):
    """
    Checks that the user has called BOSS properly by examining the arguments
    given, number of files and filename extensions. BOSS should be called with
    one of the following:
        boss o options/.rst-file
        boss op options/.rst-file
        boss p .rst-file .out-file
        boss m .rst-file local_minima.dat
    """
    # TODO prevent calling boss pm
    some_args = (len(args) > 0)
    if some_args:
        optim_ok = ('o' in args[0] and len(args) == 2)
        optim_ok = (optim_ok or 's' == args[0])
        justpp_arg_ok = ('o' not in args[0] and 'p' in args[0] and
                         len(args) == 3)
        mep_arg_ok = ('o' not in args[0] and 'm' in args[0] and
                      len(args) == 3)
        rst_incl = (len(args) >= 2 and '.rst' in args[1])
        out_incl = (len(args) == 3 and '.out' in args[2])
        dat_incl = (len(args) == 3 and '.dat' in args[2])
        justpp_ok = (justpp_arg_ok and rst_incl and out_incl)
        mep_ok = (mep_arg_ok and rst_incl and dat_incl)
    return(some_args and (optim_ok or justpp_ok or mep_ok))


def files_ok(filenames):
    """
    Checks that the given files exist and can be opened.
    """
    for fname in filenames:
        try:
            f = open(fname, 'r')
            f.close()
        except FileNotFoundError:
            print("Could not find file '" + fname + "'")
            return(False)
    return(True)


# Start BOSS
if __name__ == "__main__":
    main()
