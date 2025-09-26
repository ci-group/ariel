from _typeshed import Incomplete
from ctypes import windll as windll

__version__: str
__version_info__: Incomplete
unicode = str
system: Incomplete

def user_data_dir(appname=None, appauthor=None, version=None, roaming: bool = False):
    '''Return full path to the user-specific data dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "roaming" (boolean, default False) can be set True to use the Windows
            roaming appdata directory. That means that for users on a Windows
            network setup for roaming profiles, this user data will be
            sync\'d on login. See
            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>
            for a discussion of issues.

    Typical user data directories are:
        Mac OS X:               ~/Library/Application Support/<AppName>
        Unix:                   ~/.local/share/<AppName>    # or in $XDG_DATA_HOME, if defined
        Win XP (not roaming):   C:\\Documents and Settings\\<username>\\Application Data\\<AppAuthor>\\<AppName>
        Win XP (roaming):       C:\\Documents and Settings\\<username>\\Local Settings\\Application Data\\<AppAuthor>\\<AppName>
        Win 7  (not roaming):   C:\\Users\\<username>\\AppData\\Local\\<AppAuthor>\\<AppName>
        Win 7  (roaming):       C:\\Users\\<username>\\AppData\\Roaming\\<AppAuthor>\\<AppName>

    For Unix, we follow the XDG spec and support $XDG_DATA_HOME.
    That means, by default "~/.local/share/<AppName>".
    '''
def site_data_dir(appname=None, appauthor=None, version=None, multipath: bool = False):
    '''Return full path to the user-shared data dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "multipath" is an optional parameter only applicable to *nix
            which indicates that the entire list of data dirs should be
            returned. By default, the first item from XDG_DATA_DIRS is
            returned, or \'/usr/local/share/<AppName>\',
            if XDG_DATA_DIRS is not set

    Typical site data directories are:
        Mac OS X:   /Library/Application Support/<AppName>
        Unix:       /usr/local/share/<AppName> or /usr/share/<AppName>
        Win XP:     C:\\Documents and Settings\\All Users\\Application Data\\<AppAuthor>\\<AppName>
        Vista:      (Fail! "C:\\ProgramData" is a hidden *system* directory on Vista.)
        Win 7:      C:\\ProgramData\\<AppAuthor>\\<AppName>   # Hidden, but writeable on Win 7.

    For Unix, this is using the $XDG_DATA_DIRS[0] default.

    WARNING: Do not use this on Windows. See the Vista-Fail note above for why.
    '''
def user_config_dir(appname=None, appauthor=None, version=None, roaming: bool = False):
    '''Return full path to the user-specific config dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "roaming" (boolean, default False) can be set True to use the Windows
            roaming appdata directory. That means that for users on a Windows
            network setup for roaming profiles, this user data will be
            sync\'d on login. See
            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>
            for a discussion of issues.

    Typical user config directories are:
        Mac OS X:               ~/Library/Preferences/<AppName>
        Unix:                   ~/.config/<AppName>     # or in $XDG_CONFIG_HOME, if defined
        Win *:                  same as user_data_dir

    For Unix, we follow the XDG spec and support $XDG_CONFIG_HOME.
    That means, by default "~/.config/<AppName>".
    '''
def site_config_dir(appname=None, appauthor=None, version=None, multipath: bool = False):
    '''Return full path to the user-shared data dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "multipath" is an optional parameter only applicable to *nix
            which indicates that the entire list of config dirs should be
            returned. By default, the first item from XDG_CONFIG_DIRS is
            returned, or \'/etc/xdg/<AppName>\', if XDG_CONFIG_DIRS is not set

    Typical site config directories are:
        Mac OS X:   same as site_data_dir
        Unix:       /etc/xdg/<AppName> or $XDG_CONFIG_DIRS[i]/<AppName> for each value in
                    $XDG_CONFIG_DIRS
        Win *:      same as site_data_dir
        Vista:      (Fail! "C:\\ProgramData" is a hidden *system* directory on Vista.)

    For Unix, this is using the $XDG_CONFIG_DIRS[0] default, if multipath=False

    WARNING: Do not use this on Windows. See the Vista-Fail note above for why.
    '''
def user_cache_dir(appname=None, appauthor=None, version=None, opinion: bool = True):
    '''Return full path to the user-specific cache dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "opinion" (boolean) can be False to disable the appending of
            "Cache" to the base app data dir for Windows. See
            discussion below.

    Typical user cache directories are:
        Mac OS X:   ~/Library/Caches/<AppName>
        Unix:       ~/.cache/<AppName> (XDG default)
        Win XP:     C:\\Documents and Settings\\<username>\\Local Settings\\Application Data\\<AppAuthor>\\<AppName>\\Cache
        Vista:      C:\\Users\\<username>\\AppData\\Local\\<AppAuthor>\\<AppName>\\Cache

    On Windows the only suggestion in the MSDN docs is that local settings go in
    the `CSIDL_LOCAL_APPDATA` directory. This is identical to the non-roaming
    app data dir (the default returned by `user_data_dir` above). Apps typically
    put cache data somewhere *under* the given dir here. Some examples:
        ...\\Mozilla\\Firefox\\Profiles\\<ProfileName>\\Cache
        ...\\Acme\\SuperApp\\Cache\\1.0
    OPINION: This function appends "Cache" to the `CSIDL_LOCAL_APPDATA` value.
    This can be disabled with the `opinion=False` option.
    '''
def user_state_dir(appname=None, appauthor=None, version=None, roaming: bool = False):
    '''Return full path to the user-specific state dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "roaming" (boolean, default False) can be set True to use the Windows
            roaming appdata directory. That means that for users on a Windows
            network setup for roaming profiles, this user data will be
            sync\'d on login. See
            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>
            for a discussion of issues.

    Typical user state directories are:
        Mac OS X:  same as user_data_dir
        Unix:      ~/.local/state/<AppName>   # or in $XDG_STATE_HOME, if defined
        Win *:     same as user_data_dir

    For Unix, we follow this Debian proposal <https://wiki.debian.org/XDGBaseDirectorySpecification#state>
    to extend the XDG spec and support $XDG_STATE_HOME.

    That means, by default "~/.local/state/<AppName>".
    '''
def user_log_dir(appname=None, appauthor=None, version=None, opinion: bool = True):
    '''Return full path to the user-specific log dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "opinion" (boolean) can be False to disable the appending of
            "Logs" to the base app data dir for Windows, and "log" to the
            base cache dir for Unix. See discussion below.

    Typical user log directories are:
        Mac OS X:   ~/Library/Logs/<AppName>
        Unix:       ~/.cache/<AppName>/log  # or under $XDG_CACHE_HOME if defined
        Win XP:     C:\\Documents and Settings\\<username>\\Local Settings\\Application Data\\<AppAuthor>\\<AppName>\\Logs
        Vista:      C:\\Users\\<username>\\AppData\\Local\\<AppAuthor>\\<AppName>\\Logs

    On Windows the only suggestion in the MSDN docs is that local settings
    go in the `CSIDL_LOCAL_APPDATA` directory. (Note: I\'m interested in
    examples of what some windows apps use for a logs dir.)

    OPINION: This function appends "Logs" to the `CSIDL_LOCAL_APPDATA`
    value for Windows and appends "log" to the user cache dir for Unix.
    This can be disabled with the `opinion=False` option.
    '''

class AppDirs:
    """Convenience wrapper for getting application dirs."""
    appname: Incomplete
    appauthor: Incomplete
    version: Incomplete
    roaming: Incomplete
    multipath: Incomplete
    def __init__(self, appname=None, appauthor=None, version=None, roaming: bool = False, multipath: bool = False) -> None: ...
    @property
    def user_data_dir(self): ...
    @property
    def site_data_dir(self): ...
    @property
    def user_config_dir(self): ...
    @property
    def site_config_dir(self): ...
    @property
    def user_cache_dir(self): ...
    @property
    def user_state_dir(self): ...
    @property
    def user_log_dir(self): ...

def _get_win_folder_from_registry(csidl_name):
    """This is a fallback technique at best. I'm not sure if using the
    registry for this guarantees us the correct answer for all CSIDL_*
    names.
    """
def _get_win_folder_with_pywin32(csidl_name): ...
def _get_win_folder_with_ctypes(csidl_name): ...
def _get_win_folder_with_jna(csidl_name): ...
_get_win_folder = _get_win_folder_with_pywin32
_get_win_folder = _get_win_folder_with_ctypes
_get_win_folder = _get_win_folder_with_jna
_get_win_folder = _get_win_folder_from_registry
