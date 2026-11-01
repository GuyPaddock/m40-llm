#!/usr/bin/env perl
use strict;
use warnings;
use Digest::SHA qw(sha256_hex);
use JSON::PP;

sub read_bytes {
    open my $file, '<', $_[0] or die "$!";
    local $/;
    return <$file>;
}

sub read_functions {
    my ($path) = @_;
    my (%functions, $name);
    for my $line (split /\n/, read_bytes($path)) {
        $name = $1 if $line =~ /Function : (\S+)/;
        next unless defined $name;
        while ($line =~ /0x([a-f0-9]{16})/g) {
            push @{$functions{$name}}, "0x$1\n";
        }
    }
    return \%functions;
}

my ($old_path, $new_path, $resources_path) = @ARGV;
die "provide reference SASS, candidate SASS and resources\n" unless defined $resources_path;
my $before = read_functions($old_path);
my $after = read_functions($new_path);
my $candidate = '_ZN13m40llm_q8_ffn23gate_up_joint_four_rowsEPKfPKjPK5uint2PfS7_S7_';
die 'expected 137 original controls and one candidate' unless keys(%$before)==137 && keys(%$after)==138;
my @original;
for my $name (sort keys %$before) {
    die "missing $name" unless exists $after->{$name};
    my $old = join '', @{$before->{$name}};
    my $new = join '', @{$after->{$name}};
    die "changed control $name" unless $old eq $new;
    push @original, {name=>$name, words=>scalar @{$after->{$name}}, sha256=>sha256_hex($new)};
}
my @added = grep {!exists $before->{$_}} keys %$after;
die 'unapproved candidate' unless @added==1 && $added[0] eq $candidate;
my $resources = read_bytes($resources_path);
$resources =~ /Function \Q$candidate\E:\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)/
    or die 'missing candidate resources';
my ($registers,$stack,$shared,$local)=($1,$2,$3,$4);
die 'candidate resource ceiling failed' unless $registers<=64 && $stack==0 && $shared==0 && $local==0;
my $words = join '', @{$after->{$candidate}};
print JSON::PP->new->canonical->pretty->encode({
    is_passed=>JSON::PP::true, original_count=>137, original_entries=>\@original,
    candidate=>{name=>$candidate,words=>scalar @{$after->{$candidate}},sha256=>sha256_hex($words),
        registers_per_thread=>0+$registers,stack_bytes=>0+$stack,static_shared_bytes=>0+$shared,
        local_bytes=>0+$local,dynamic_shared_bytes=>4608,threads=>128},
    scope=>'instruction identity and static resources, not achieved occupancy or DRAM traffic'});
